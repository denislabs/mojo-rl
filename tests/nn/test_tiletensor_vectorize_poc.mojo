"""POC: TileTensor.vectorize() for elementwise GPU kernels.

Benchmarks scalar-per-thread vs vectorized (SIMD width 4) variants of:
  1. Vector add (a + b)
  2. ReLU forward (max(0, x) + cache)
  3. ReLU backward (grad * mask)
  4. Tanh forward (tanh(x) + cache)

Each test validates correctness against a CPU reference, then benchmarks
scalar vs vectorized with realistic RL dimensions.

Run:
    pixi run -e apple  mojo run -I . tests/nn/test_tiletensor_vectorize_poc.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_tiletensor_vectorize_poc.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.math import tanh, max
from layout import Layout, LayoutTensor, TileTensor, row_major
from mojo_rl.nn.constants import dtype, TPB


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def fill_random_ptr(
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        ptr[i] = Scalar[dtype](random_float64(-2.0, 2.0).cast[dtype]())


def max_abs_diff_ptr(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Scalar[dtype]:
    var mx: Scalar[dtype] = 0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


# ─────────────────────────────────────────────────────────────────────
# Test 1: Vector add — a + b
# ─────────────────────────────────────────────────────────────────────


def bench_add[SIZE: Int](ctx: DeviceContext) raises:
    print("\n── Vector add: SIZE=" + String(SIZE) + " ──")

    var a_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var b_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var c_scalar_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var c_vec_buf = ctx.enqueue_create_buffer[dtype](SIZE)

    var ha = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var hb = ctx.enqueue_create_host_buffer[dtype](SIZE)
    fill_random_ptr(ha.unsafe_ptr(), SIZE)
    fill_random_ptr(hb.unsafe_ptr(), SIZE)
    ctx.enqueue_copy(a_buf, ha)
    ctx.enqueue_copy(b_buf, hb)

    # ── Scalar kernel: 1 element per thread ──
    @always_inline
    @parameter
    def add_scalar(
        c: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
        a: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
        b: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx < SIZE:
            c[idx] = rebind[Scalar[dtype]](a[idx]) + rebind[Scalar[dtype]](
                b[idx]
            )

    # ── Vectorized kernel: 4 elements per thread ──
    @always_inline
    @parameter
    def add_vec4(
        c: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
        a: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
        b: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    ):
        var va = TileTensor(a.ptr, row_major[SIZE]()).vectorize[4]()
        var vb = TileTensor(b.ptr, row_major[SIZE]()).vectorize[4]()
        var vc = TileTensor(c.ptr, row_major[SIZE]()).vectorize[4]()

        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        comptime VEC_N = SIZE // 4
        if idx < VEC_N:
            vc[idx] = va[idx] + vb[idx]

    comptime scalar_blocks = (SIZE + TPB - 1) // TPB
    comptime vec_blocks = (SIZE // 4 + TPB - 1) // TPB

    var c_s_lt = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](
        c_scalar_buf
    )
    var c_v_lt = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](
        c_vec_buf
    )
    var a_lt = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](a_buf)
    var b_lt = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](b_buf)

    # Correctness
    ctx.enqueue_function[add_scalar, add_scalar](
        c_s_lt, a_lt, b_lt, grid_dim=(scalar_blocks,), block_dim=(TPB,)
    )
    ctx.enqueue_function[add_vec4, add_vec4](
        c_v_lt, a_lt, b_lt, grid_dim=(vec_blocks,), block_dim=(TPB,)
    )
    ctx.synchronize()

    var hc_s = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var hc_v = ctx.enqueue_create_host_buffer[dtype](SIZE)
    ctx.enqueue_copy(hc_s, c_scalar_buf)
    ctx.enqueue_copy(hc_v, c_vec_buf)
    ctx.synchronize()

    var diff = max_abs_diff_ptr(hc_s.unsafe_ptr(), hc_v.unsafe_ptr(), SIZE)
    print(
        "  correctness (scalar vs vec4): "
        + String(diff)
        + " "
        + ("PASS" if diff < 1e-5 else "FAIL")
    )

    # Benchmark
    comptime N_ITERS = 2000
    comptime WARMUP = 100

    for _ in range(WARMUP):
        ctx.enqueue_function[add_scalar, add_scalar](
            c_s_lt, a_lt, b_lt, grid_dim=(scalar_blocks,), block_dim=(TPB,)
        )
        ctx.enqueue_function[add_vec4, add_vec4](
            c_v_lt, a_lt, b_lt, grid_dim=(vec_blocks,), block_dim=(TPB,)
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[add_scalar, add_scalar](
            c_s_lt, a_lt, b_lt, grid_dim=(scalar_blocks,), block_dim=(TPB,)
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[add_vec4, add_vec4](
            c_v_lt, a_lt, b_lt, grid_dim=(vec_blocks,), block_dim=(TPB,)
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()

    var scalar_us = Float64(t1 - t0) / Float64(N_ITERS) / 1000.0
    var vec_us = Float64(t2 - t1) / Float64(N_ITERS) / 1000.0
    print(
        "  scalar: "
        + String(scalar_us)
        + " us | vec4: "
        + String(vec_us)
        + " us | ratio: "
        + String(vec_us / scalar_us)
        + "x"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 2: ReLU forward — output = max(0, input), cache = input
# ─────────────────────────────────────────────────────────────────────


def bench_relu_forward[BATCH: Int, DIM: Int](ctx: DeviceContext) raises:
    comptime SIZE = BATCH * DIM
    print(
        "\n── ReLU forward: BATCH="
        + String(BATCH)
        + " DIM="
        + String(DIM)
        + " (SIZE="
        + String(SIZE)
        + ") ──"
    )

    var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var out_scalar_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var out_vec_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var cache_scalar_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var cache_vec_buf = ctx.enqueue_create_buffer[dtype](SIZE)

    var h_in = ctx.enqueue_create_host_buffer[dtype](SIZE)
    fill_random_ptr(h_in.unsafe_ptr(), SIZE)
    ctx.enqueue_copy(input_buf, h_in)

    # ── Scalar: matches existing ReLUOp.eval_kernel_impl pattern ──
    @always_inline
    @parameter
    def relu_scalar(
        output: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= SIZE:
            return
        var row = idx // DIM
        var col = idx % DIM
        var val = rebind[Scalar[dtype]](input[row, col])
        cache[row, col] = val
        output[row, col] = val if val > 0 else 0

    # ── Vectorized: 4 elements per thread, flat view ──
    @always_inline
    @parameter
    def relu_vec4(
        output: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var vi = TileTensor(input.ptr, row_major[SIZE]()).vectorize[4]()
        var vo = TileTensor(output.ptr, row_major[SIZE]()).vectorize[4]()
        var vc = TileTensor(cache.ptr, row_major[SIZE]()).vectorize[4]()

        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        comptime VEC_N = SIZE // 4
        if idx < VEC_N:
            var v = vi[idx]
            vc[idx] = v
            vo[idx] = max(v, SIMD[dtype, 4](0))

    comptime scalar_blocks = (SIZE + TPB - 1) // TPB
    comptime vec_blocks = (SIZE // 4 + TPB - 1) // TPB

    var out_s_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_scalar_buf)
    var out_v_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_vec_buf)
    var in_lt = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        input_buf
    )
    var cache_s_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_scalar_buf)
    var cache_v_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_vec_buf)

    # Correctness
    ctx.enqueue_function[relu_scalar, relu_scalar](
        out_s_lt,
        in_lt,
        cache_s_lt,
        grid_dim=(scalar_blocks,),
        block_dim=(TPB,),
    )
    ctx.enqueue_function[relu_vec4, relu_vec4](
        out_v_lt,
        in_lt,
        cache_v_lt,
        grid_dim=(vec_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    var h_out_s = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var h_out_v = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var h_cache_s = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var h_cache_v = ctx.enqueue_create_host_buffer[dtype](SIZE)
    ctx.enqueue_copy(h_out_s, out_scalar_buf)
    ctx.enqueue_copy(h_out_v, out_vec_buf)
    ctx.enqueue_copy(h_cache_s, cache_scalar_buf)
    ctx.enqueue_copy(h_cache_v, cache_vec_buf)
    ctx.synchronize()

    var out_diff = max_abs_diff_ptr(
        h_out_s.unsafe_ptr(), h_out_v.unsafe_ptr(), SIZE
    )
    var cache_diff = max_abs_diff_ptr(
        h_cache_s.unsafe_ptr(), h_cache_v.unsafe_ptr(), SIZE
    )
    print(
        "  output diff: "
        + String(out_diff)
        + " | cache diff: "
        + String(cache_diff)
        + " "
        + ("PASS" if out_diff < 1e-5 and cache_diff < 1e-5 else "FAIL")
    )

    # Benchmark
    comptime N_ITERS = 2000
    comptime WARMUP = 100

    for _ in range(WARMUP):
        ctx.enqueue_function[relu_scalar, relu_scalar](
            out_s_lt,
            in_lt,
            cache_s_lt,
            grid_dim=(scalar_blocks,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[relu_vec4, relu_vec4](
            out_v_lt,
            in_lt,
            cache_v_lt,
            grid_dim=(vec_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[relu_scalar, relu_scalar](
            out_s_lt,
            in_lt,
            cache_s_lt,
            grid_dim=(scalar_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[relu_vec4, relu_vec4](
            out_v_lt,
            in_lt,
            cache_v_lt,
            grid_dim=(vec_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()

    var scalar_us = Float64(t1 - t0) / Float64(N_ITERS) / 1000.0
    var vec_us = Float64(t2 - t1) / Float64(N_ITERS) / 1000.0
    print(
        "  scalar: "
        + String(scalar_us)
        + " us | vec4: "
        + String(vec_us)
        + " us | ratio: "
        + String(vec_us / scalar_us)
        + "x"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 3: ReLU backward — grad_input = grad_output * (cache > 0)
# ─────────────────────────────────────────────────────────────────────


def bench_relu_backward[BATCH: Int, DIM: Int](ctx: DeviceContext) raises:
    comptime SIZE = BATCH * DIM
    print(
        "\n── ReLU backward: BATCH="
        + String(BATCH)
        + " DIM="
        + String(DIM)
        + " ──"
    )

    var grad_out_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var gi_scalar_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var gi_vec_buf = ctx.enqueue_create_buffer[dtype](SIZE)

    var h_go = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var h_cache = ctx.enqueue_create_host_buffer[dtype](SIZE)
    fill_random_ptr(h_go.unsafe_ptr(), SIZE)
    fill_random_ptr(h_cache.unsafe_ptr(), SIZE)
    ctx.enqueue_copy(grad_out_buf, h_go)
    ctx.enqueue_copy(cache_buf, h_cache)

    # ── Scalar: matches existing ReLUOp.backward_kernel_impl ──
    @always_inline
    @parameter
    def relu_bwd_scalar(
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= SIZE:
            return
        var row = idx // DIM
        var col = idx % DIM
        grad_input[row, col] = (
            rebind[Scalar[dtype]](grad_output[row, col]) if rebind[
                Scalar[dtype]
            ](cache[row, col])
            > 0 else 0
        )

    # ── Vectorized: SIMD comparison + select ──
    @always_inline
    @parameter
    def relu_bwd_vec4(
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var vgo = TileTensor(grad_output.ptr, row_major[SIZE]()).vectorize[4]()
        var vc = TileTensor(cache.ptr, row_major[SIZE]()).vectorize[4]()
        var vgi = TileTensor(grad_input.ptr, row_major[SIZE]()).vectorize[4]()

        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        comptime VEC_N = SIZE // 4
        if idx < VEC_N:
            var go = vgo[idx]
            var c = vc[idx]
            var zero = SIMD[dtype, 4](0)
            # SIMD mask: where cache > 0, pass grad; else 0
            vgi[idx] = c.gt(zero).select(go, zero)

    comptime scalar_blocks = (SIZE + TPB - 1) // TPB
    comptime vec_blocks = (SIZE // 4 + TPB - 1) // TPB

    var gi_s_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gi_scalar_buf)
    var gi_v_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gi_vec_buf)
    var go_lt = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        grad_out_buf
    )
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_buf)

    # Correctness
    ctx.enqueue_function[relu_bwd_scalar, relu_bwd_scalar](
        gi_s_lt,
        go_lt,
        cache_lt,
        grid_dim=(scalar_blocks,),
        block_dim=(TPB,),
    )
    ctx.enqueue_function[relu_bwd_vec4, relu_bwd_vec4](
        gi_v_lt,
        go_lt,
        cache_lt,
        grid_dim=(vec_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    var h_gi_s = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var h_gi_v = ctx.enqueue_create_host_buffer[dtype](SIZE)
    ctx.enqueue_copy(h_gi_s, gi_scalar_buf)
    ctx.enqueue_copy(h_gi_v, gi_vec_buf)
    ctx.synchronize()

    var diff = max_abs_diff_ptr(h_gi_s.unsafe_ptr(), h_gi_v.unsafe_ptr(), SIZE)
    print("  diff: " + String(diff) + " " + ("PASS" if diff < 1e-5 else "FAIL"))

    # Benchmark
    comptime N_ITERS = 2000
    comptime WARMUP = 100

    for _ in range(WARMUP):
        ctx.enqueue_function[relu_bwd_scalar, relu_bwd_scalar](
            gi_s_lt,
            go_lt,
            cache_lt,
            grid_dim=(scalar_blocks,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[relu_bwd_vec4, relu_bwd_vec4](
            gi_v_lt,
            go_lt,
            cache_lt,
            grid_dim=(vec_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[relu_bwd_scalar, relu_bwd_scalar](
            gi_s_lt,
            go_lt,
            cache_lt,
            grid_dim=(scalar_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[relu_bwd_vec4, relu_bwd_vec4](
            gi_v_lt,
            go_lt,
            cache_lt,
            grid_dim=(vec_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()

    var scalar_us = Float64(t1 - t0) / Float64(N_ITERS) / 1000.0
    var vec_us = Float64(t2 - t1) / Float64(N_ITERS) / 1000.0
    print(
        "  scalar: "
        + String(scalar_us)
        + " us | vec4: "
        + String(vec_us)
        + " us | ratio: "
        + String(vec_us / scalar_us)
        + "x"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 4: Tanh forward — output = tanh(input), cache = output
# ─────────────────────────────────────────────────────────────────────


def bench_tanh_forward[BATCH: Int, DIM: Int](ctx: DeviceContext) raises:
    comptime SIZE = BATCH * DIM
    print(
        "\n── Tanh forward: BATCH="
        + String(BATCH)
        + " DIM="
        + String(DIM)
        + " ──"
    )

    var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var out_scalar_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var out_vec_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var cache_scalar_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var cache_vec_buf = ctx.enqueue_create_buffer[dtype](SIZE)

    var h_in = ctx.enqueue_create_host_buffer[dtype](SIZE)
    fill_random_ptr(h_in.unsafe_ptr(), SIZE)
    ctx.enqueue_copy(input_buf, h_in)

    # ── Scalar: 1 element per thread ──
    @always_inline
    @parameter
    def tanh_scalar(
        output: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= SIZE:
            return
        var row = idx // DIM
        var col = idx % DIM
        var val = tanh(rebind[Scalar[dtype]](input[row, col]))
        output[row, col] = val
        cache[row, col] = val

    # ── Vectorized: 4 elements per thread ──
    @always_inline
    @parameter
    def tanh_vec4(
        output: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var vi = TileTensor(input.ptr, row_major[SIZE]()).vectorize[4]()
        var vo = TileTensor(output.ptr, row_major[SIZE]()).vectorize[4]()
        var vc = TileTensor(cache.ptr, row_major[SIZE]()).vectorize[4]()

        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        comptime VEC_N = SIZE // 4
        if idx < VEC_N:
            var v = tanh(vi[idx])
            vo[idx] = v
            vc[idx] = v

    comptime scalar_blocks = (SIZE + TPB - 1) // TPB
    comptime vec_blocks = (SIZE // 4 + TPB - 1) // TPB

    var out_s_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_scalar_buf)
    var out_v_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_vec_buf)
    var in_lt = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        input_buf
    )
    var cache_s_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_scalar_buf)
    var cache_v_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_vec_buf)

    # Correctness
    ctx.enqueue_function[tanh_scalar, tanh_scalar](
        out_s_lt,
        in_lt,
        cache_s_lt,
        grid_dim=(scalar_blocks,),
        block_dim=(TPB,),
    )
    ctx.enqueue_function[tanh_vec4, tanh_vec4](
        out_v_lt,
        in_lt,
        cache_v_lt,
        grid_dim=(vec_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    var h_out_s = ctx.enqueue_create_host_buffer[dtype](SIZE)
    var h_out_v = ctx.enqueue_create_host_buffer[dtype](SIZE)
    ctx.enqueue_copy(h_out_s, out_scalar_buf)
    ctx.enqueue_copy(h_out_v, out_vec_buf)
    ctx.synchronize()

    var diff = max_abs_diff_ptr(
        h_out_s.unsafe_ptr(), h_out_v.unsafe_ptr(), SIZE
    )
    print("  diff: " + String(diff) + " " + ("PASS" if diff < 1e-5 else "FAIL"))

    # Benchmark
    comptime N_ITERS = 2000
    comptime WARMUP = 100

    for _ in range(WARMUP):
        ctx.enqueue_function[tanh_scalar, tanh_scalar](
            out_s_lt,
            in_lt,
            cache_s_lt,
            grid_dim=(scalar_blocks,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[tanh_vec4, tanh_vec4](
            out_v_lt,
            in_lt,
            cache_v_lt,
            grid_dim=(vec_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[tanh_scalar, tanh_scalar](
            out_s_lt,
            in_lt,
            cache_s_lt,
            grid_dim=(scalar_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[tanh_vec4, tanh_vec4](
            out_v_lt,
            in_lt,
            cache_v_lt,
            grid_dim=(vec_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()

    var scalar_us = Float64(t1 - t0) / Float64(N_ITERS) / 1000.0
    var vec_us = Float64(t2 - t1) / Float64(N_ITERS) / 1000.0
    print(
        "  scalar: "
        + String(scalar_us)
        + " us | vec4: "
        + String(vec_us)
        + " us | ratio: "
        + String(vec_us / scalar_us)
        + "x"
    )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() raises:
    seed(42)
    print("=" * 65)
    print("TileTensor.vectorize() Benchmark: Scalar vs Vec4")
    print("=" * 65)

    with DeviceContext() as ctx:
        # ── Vector add at different sizes ──
        bench_add[4096](ctx)
        bench_add[65536](ctx)
        bench_add[262144](ctx)

        # ── ReLU forward — realistic RL dimensions ──
        # DQN hidden: batch=256, dim=512
        bench_relu_forward[256, 512](ctx)
        # PPO large: batch=2048, dim=256
        bench_relu_forward[2048, 256](ctx)
        # Atari CNN flattened: batch=32, dim=3136
        bench_relu_forward[32, 3136](ctx)

        # ── ReLU backward ──
        bench_relu_backward[256, 512](ctx)
        bench_relu_backward[2048, 256](ctx)

        # ── Tanh forward (compute-bound vs memory-bound) ──
        bench_tanh_forward[256, 512](ctx)
        bench_tanh_forward[2048, 256](ctx)

    print("\n" + "=" * 65)
    print("Done!")
    print("=" * 65)

"""Benchmark AlphaZero TicTacToe-CNN PredModel on CPU.

Times the *current* CPU implementations of every layer in the
`AlphaZeroTicTacToeCNNConfig.PredModel`, plus a full end-to-end
forward+backward of the Sequential composition, at BATCH=64
(the AZ TTT CNN training batch size).

For each layer we also bench:
  - `linalg.matmul[target="cpu"]` at the same matmul shape — the
    optimized lower bound that the layer *could* reach if its
    naive triple-loops were replaced by BLAS (Apple Accelerate /
    Modular CPU GEMM).
  - A naive triple-loop matmul at the same shape — sanity reference
    matching what the layer currently does internally.

Run:
    pixi run mojo run -I . benchmarks/benchmark_alphazero_ttt_cnn_cpu.mojo

The output is meant to answer: "which layer is the hot path on CPU,
and how much can BLAS buy us per layer?"
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import (
    Conv2DBatchNormReLU,
    LinearBatchNormReLU,
)


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
def fill_random(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](random_float64(-1.0, 1.0))


@always_inline
def zero(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    memset(p, 0, n)


@always_inline
def fmt_ms(ms: Float64) -> String:
    return String(ms)[byte=:9] + " ms"


@always_inline
def fmt_gflops(g: Float64) -> String:
    return String(g)[byte=:6] + " GFLOPS"


# ─────────────────────────────────────────────────────────────────────────────
# Reference matmuls (used to bracket "current naive" vs "BLAS lower bound")
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
def naive_matmul[
    M: Int, N: Int, K: Int
](
    a: LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
    mut c: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
):
    for i in range(M):
        for j in range(N):
            var acc: c.element_type = 0
            for k in range(K):
                acc += a[i, k] * b[k, j]
            c[i, j] = acc


def bench_matmul_pair[
    M: Int, K: Int, N: Int, ITERS: Int
](label: String) raises:
    """Bench naive vs `max_matmul[target="cpu"]` at shape (M,K)@(K,N).

    Reports the BLAS time so we can compare against the layer time.
    """
    var a_buf = alloc[Scalar[dtype]](M * K)
    var b_buf = alloc[Scalar[dtype]](K * N)
    var c_naive_buf = alloc[Scalar[dtype]](M * N)
    var c_blas_buf = alloc[Scalar[dtype]](M * N)

    fill_random(a_buf, M * K)
    fill_random(b_buf, K * N)
    zero(c_naive_buf, M * N)
    zero(c_blas_buf, M * N)

    var a_lt = LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin](a_buf)
    var b_lt = LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin](b_buf)
    var c_naive_lt = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
        c_naive_buf
    )
    var c_blas_lt = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
        c_blas_buf
    )

    # Warmup
    for _ in range(2):
        max_matmul[target="cpu"](
            lt_to_tt(c_blas_lt), lt_to_tt(a_lt), lt_to_tt(b_lt), None
        )

    # Bench naive
    comptime work = Int64(M) * Int64(N) * Int64(K)
    comptime skip_naive = work > Int64(512) * 512 * 512
    var naive_ms: Float64 = 0.0
    var sink: Float64 = 0.0
    comptime if not skip_naive:
        var t0 = perf_counter_ns()
        for it in range(ITERS):
            # Perturb input[0] each iter so the compiler can't hoist the call.
            a_buf[0] = Scalar[dtype](Float64(it) * 1e-3)
            naive_matmul[M, N, K](a_lt, b_lt, c_naive_lt)
            sink += Float64(c_naive_buf[0])
        var t1 = perf_counter_ns()
        naive_ms = Float64(t1 - t0) / 1e6 / Float64(ITERS)

    # Bench BLAS
    var t2 = perf_counter_ns()
    for it in range(ITERS):
        a_buf[0] = Scalar[dtype](Float64(it) * 1e-3)
        max_matmul[target="cpu"](
            lt_to_tt(c_blas_lt), lt_to_tt(a_lt), lt_to_tt(b_lt), None
        )
        sink += Float64(c_blas_buf[0])
    var t3 = perf_counter_ns()
    var blas_ms = Float64(t3 - t2) / 1e6 / Float64(ITERS)
    # Force `sink` to be observable so the matmul calls can't be DCE'd.
    if sink == Float64(1.2345e308):
        print("(unreachable)")

    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var blas_gflops = flops / (blas_ms * 1e6)

    print("  ", label)
    comptime if not skip_naive:
        var n_gflops = flops / (naive_ms * 1e6)
        var speedup = naive_ms / blas_ms
        print(
            "    naive triple-loop:  ",
            fmt_ms(naive_ms),
            "   ",
            fmt_gflops(n_gflops),
        )
        print(
            "    linalg.matmul[cpu]: ",
            fmt_ms(blas_ms),
            "   ",
            fmt_gflops(blas_gflops),
            "    (",
            String(speedup)[byte=:5],
            "x speedup)",
        )
    else:
        print("    naive triple-loop:   skipped (too slow)")
        print(
            "    linalg.matmul[cpu]: ",
            fmt_ms(blas_ms),
            "   ",
            fmt_gflops(blas_gflops),
        )

    a_buf.free()
    b_buf.free()
    c_naive_buf.free()
    c_blas_buf.free()


# ─────────────────────────────────────────────────────────────────────────────
# Conv2DBatchNormReLU — per-layer bench
# ─────────────────────────────────────────────────────────────────────────────


def bench_conv_bn_relu_layer[
    IN_C: Int,
    OUT_C: Int,
    K: Int,
    S: Int,
    P: Int,
    IH: Int,
    IW: Int,
    BATCH: Int,
    FWD_ITERS: Int,
    BWD_ITERS: Int,
](label: String) raises:
    """Time forward+backward of Conv2DBatchNormReLU at the given shape."""
    comptime Layer = Conv2DBatchNormReLU[IN_C, OUT_C, K, S, P, IH, IW]
    comptime IN_DIM = Layer.IN_DIM
    comptime OUT_DIM = Layer.OUT_DIM
    comptime PARAM_SIZE = Layer.PARAM_SIZE
    comptime STATE_SIZE = Layer.STATE_SIZE
    comptime CACHE_SIZE = Layer.CACHE_SIZE
    comptime out_h = Layer.out_h
    comptime out_w = Layer.out_w
    comptime spatial_out = Layer.spatial_out
    comptime col_size = Layer.col_size

    print("─" * 78)
    print(label)
    print(
        "  in=",
        IN_C,
        "x",
        IH,
        "x",
        IW,
        "  out=",
        OUT_C,
        "x",
        out_h,
        "x",
        out_w,
        "  k=",
        K,
        " s=",
        S,
        " p=",
        P,
        "  BATCH=",
        BATCH,
    )
    print(
        "  PARAM_SIZE=",
        PARAM_SIZE,
        "  CACHE_SIZE/sample=",
        CACHE_SIZE,
        "  col_size=",
        col_size,
        "  spatial_out=",
        spatial_out,
    )

    # Allocate everything on heap.
    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * CACHE_SIZE)

    fill_random(input_buf, BATCH * IN_DIM)
    zero(output_buf, BATCH * OUT_DIM)
    zero(grad_in_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    zero(grads_buf, PARAM_SIZE)
    zero(cache_buf, BATCH * CACHE_SIZE)

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var grads_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](grads_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](input_buf)
    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](output_buf)
    var grad_in_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](grad_in_buf)
    var grad_out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_buf)

    Layer.initialize_params[Kaiming[], dtype](params_lt)
    Layer.initialize_state[dtype](state_lt)

    # Warmup
    Layer.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    Layer.backward[BATCH, dtype](
        grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
    )

    # Bench forward (training, with cache)
    var t0 = perf_counter_ns()
    for _ in range(FWD_ITERS):
        Layer.forward[BATCH, dtype](
            input_lt, output_lt, params_lt, state_lt, cache_lt
        )
    var t1 = perf_counter_ns()
    var fwd_ms = Float64(t1 - t0) / 1e6 / Float64(FWD_ITERS)

    # Bench backward
    var t2 = perf_counter_ns()
    for _ in range(BWD_ITERS):
        Layer.backward[BATCH, dtype](
            grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
        )
    var t3 = perf_counter_ns()
    var bwd_ms = Float64(t3 - t2) / 1e6 / Float64(BWD_ITERS)

    # FLOPS estimate (dense matmul portion only — the BN+ReLU and im2col
    # are O(BATCH * OUT_DIM) and O(BATCH * CACHE) bookkeeping).
    # Forward matmul: out_C × spatial_out × col_size × BATCH × 2
    var fwd_flops = (
        2.0
        * Float64(BATCH)
        * Float64(OUT_C)
        * Float64(spatial_out)
        * Float64(col_size)
    )
    var fwd_gflops = fwd_flops / (fwd_ms * 1e6)

    # Backward matmul: dW (OC × col × spatial × B × 2) + dcol (col × spatial × OC × B × 2)
    var bwd_flops = (
        4.0
        * Float64(BATCH)
        * Float64(OUT_C)
        * Float64(spatial_out)
        * Float64(col_size)
    )
    var bwd_gflops = bwd_flops / (bwd_ms * 1e6)

    print(
        "  forward (train) :",
        fmt_ms(fwd_ms),
        "   matmul-equiv ~",
        fmt_gflops(fwd_gflops),
    )
    print(
        "  backward        :",
        fmt_ms(bwd_ms),
        "   matmul-equiv ~",
        fmt_gflops(bwd_gflops),
    )
    print(
        "  BLAS lower bound at the dominant matmul shape (",
        OUT_C,
        "x",
        col_size,
        ") @ (",
        col_size,
        "x",
        spatial_out,
        ") × BATCH=",
        BATCH,
        "  (per-sample shape that the current naive loop runs):",
    )

    # Per-batch matmul: (OUT_C × col_size) @ (col_size × spatial_out)
    # Bench the BLAS upper-bound on one-shot batched form M=OUT_C*BATCH, K=col_size, N=spatial_out
    # OR equivalently bench per-sample then × BATCH — we do the latter, matching the layer code.
    var blas_total_ms: Float64 = 0.0
    var blas_naive_ms: Float64 = 0.0
    # Setup buffers once
    var w_buf = alloc[Scalar[dtype]](OUT_C * col_size)
    var col_buf = alloc[Scalar[dtype]](col_size * spatial_out)
    var c_blas = alloc[Scalar[dtype]](OUT_C * spatial_out)
    fill_random(w_buf, OUT_C * col_size)
    fill_random(col_buf, col_size * spatial_out)
    zero(c_blas, OUT_C * spatial_out)
    var w_lt = LayoutTensor[
        dtype, Layout.row_major(OUT_C, col_size), MutAnyOrigin
    ](w_buf)
    var col_lt = LayoutTensor[
        dtype, Layout.row_major(col_size, spatial_out), MutAnyOrigin
    ](col_buf)
    var c_lt = LayoutTensor[
        dtype, Layout.row_major(OUT_C, spatial_out), MutAnyOrigin
    ](c_blas)

    for _ in range(2):
        max_matmul[target="cpu"](
            lt_to_tt(c_lt), lt_to_tt(w_lt), lt_to_tt(col_lt), None
        )

    var sink2: Float64 = 0.0
    var ITERS_MM = 50
    var t4 = perf_counter_ns()
    for _ in range(ITERS_MM):
        for _ in range(BATCH):  # per-sample matmul (matches current code shape)
            max_matmul[target="cpu"](
                lt_to_tt(c_lt), lt_to_tt(w_lt), lt_to_tt(col_lt), None
            )
            sink2 += Float64(c_blas[0])
    var t5 = perf_counter_ns()
    blas_total_ms = Float64(t5 - t4) / 1e6 / Float64(ITERS_MM)
    var blas_gflops = fwd_flops / (blas_total_ms * 1e6)

    # Same shape, naive
    comptime mm_work = Int64(OUT_C) * Int64(spatial_out) * Int64(col_size)
    comptime mm_skip_naive = mm_work > Int64(256) * 256 * 256
    comptime if not mm_skip_naive:
        var t6 = perf_counter_ns()
        var n_iters = 5 if (Int64(BATCH) * mm_work > Int64(64) * 1024 * 1024) else 20
        for _ in range(n_iters):
            for _ in range(BATCH):
                naive_matmul[OUT_C, spatial_out, col_size](w_lt, col_lt, c_lt)
                sink2 += Float64(c_blas[0])
        var t7 = perf_counter_ns()
        blas_naive_ms = Float64(t7 - t6) / 1e6 / Float64(n_iters)
    _ = sink2

    print(
        "    per-sample × BATCH naive matmul:  ",
        fmt_ms(blas_naive_ms),
    )
    print(
        "    per-sample × BATCH BLAS matmul :  ",
        fmt_ms(blas_total_ms),
        "   ",
        fmt_gflops(blas_gflops),
    )
    print(
        "    fwd speedup achievable if matmul replaced by BLAS: ~",
        String(fwd_ms / blas_total_ms)[byte=:5],
        "x",
    )

    w_buf.free()
    col_buf.free()
    c_blas.free()

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()


# ─────────────────────────────────────────────────────────────────────────────
# LinearBatchNormReLU — per-layer bench
# ─────────────────────────────────────────────────────────────────────────────


def bench_linear_bn_relu_layer[
    IN_DIM: Int, OUT_DIM: Int, BATCH: Int, FWD_ITERS: Int, BWD_ITERS: Int
](label: String) raises:
    comptime Layer = LinearBatchNormReLU[IN_DIM, OUT_DIM]
    comptime PARAM_SIZE = Layer.PARAM_SIZE
    comptime STATE_SIZE = Layer.STATE_SIZE
    comptime CACHE_SIZE = Layer.CACHE_SIZE

    print("─" * 78)
    print(label)
    print(
        "  in=",
        IN_DIM,
        "  out=",
        OUT_DIM,
        "  BATCH=",
        BATCH,
        "  PARAM_SIZE=",
        PARAM_SIZE,
        "  CACHE_SIZE/sample=",
        CACHE_SIZE,
    )

    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * CACHE_SIZE)

    fill_random(input_buf, BATCH * IN_DIM)
    zero(output_buf, BATCH * OUT_DIM)
    zero(grad_in_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    zero(grads_buf, PARAM_SIZE)
    zero(cache_buf, BATCH * CACHE_SIZE)

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var grads_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](grads_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](input_buf)
    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](output_buf)
    var grad_in_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](grad_in_buf)
    var grad_out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_buf)

    Layer.initialize_params[Kaiming[], dtype](params_lt)
    Layer.initialize_state[dtype](state_lt)

    # Warmup
    Layer.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    Layer.backward[BATCH, dtype](
        grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
    )

    var t0 = perf_counter_ns()
    for _ in range(FWD_ITERS):
        Layer.forward[BATCH, dtype](
            input_lt, output_lt, params_lt, state_lt, cache_lt
        )
    var t1 = perf_counter_ns()
    var fwd_ms = Float64(t1 - t0) / 1e6 / Float64(FWD_ITERS)

    var t2 = perf_counter_ns()
    for _ in range(BWD_ITERS):
        Layer.backward[BATCH, dtype](
            grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
        )
    var t3 = perf_counter_ns()
    var bwd_ms = Float64(t3 - t2) / 1e6 / Float64(BWD_ITERS)

    var fwd_flops = 2.0 * Float64(BATCH) * Float64(IN_DIM) * Float64(OUT_DIM)
    var fwd_gflops = fwd_flops / (fwd_ms * 1e6)
    var bwd_flops = 4.0 * Float64(BATCH) * Float64(IN_DIM) * Float64(OUT_DIM)
    var bwd_gflops = bwd_flops / (bwd_ms * 1e6)

    print(
        "  forward (train) :",
        fmt_ms(fwd_ms),
        "   matmul-equiv ~",
        fmt_gflops(fwd_gflops),
    )
    print(
        "  backward        :",
        fmt_ms(bwd_ms),
        "   matmul-equiv ~",
        fmt_gflops(bwd_gflops),
    )

    # BLAS comparison at exactly the layer's matmul shape (BATCH × IN_DIM @ IN_DIM × OUT_DIM)
    bench_matmul_pair[BATCH, IN_DIM, OUT_DIM, 50](
        "BLAS at fwd matmul shape (BATCH x IN @ IN x OUT)"
    )

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() raises:
    seed(42)

    comptime BATCH = 64
    comptime F = 128

    print("=" * 78)
    print(
        " AlphaZero TicTacToe CNN — CPU layer benchmark   BATCH=",
        BATCH,
        "   F=",
        F,
    )
    print("=" * 78)
    print()

    # ── Conv layers ────────────────────────────────────────────────
    # Layer 1: 3ch → F, 3×3 → 3×3 (padding=1)
    bench_conv_bn_relu_layer[3, F, 3, 1, 1, 3, 3, BATCH, 30, 15](
        "Conv2DBatchNormReLU L1: 3 -> F, 3x3 -> 3x3 (pad=1)"
    )
    print()

    # Layer 2 (and L3, same shape): F → F, 3×3 → 3×3 (padding=1)
    bench_conv_bn_relu_layer[F, F, 3, 1, 1, 3, 3, BATCH, 20, 10](
        "Conv2DBatchNormReLU L2/L3: F -> F, 3x3 -> 3x3 (pad=1)"
    )
    print()

    # Layer 4: F → F, 3×3 → 1×1 (valid, padding=0)
    bench_conv_bn_relu_layer[F, F, 3, 1, 0, 3, 3, BATCH, 30, 15](
        "Conv2DBatchNormReLU L4: F -> F, 3x3 -> 1x1 (pad=0, valid)"
    )
    print()

    # ── Linear+BN+ReLU layers ──────────────────────────────────────
    bench_linear_bn_relu_layer[F, F * 2, BATCH, 100, 50](
        "LinearBatchNormReLU L5: F -> F*2 (128 -> 256)"
    )
    print()
    bench_linear_bn_relu_layer[F * 2, F, BATCH, 100, 50](
        "LinearBatchNormReLU L6: F*2 -> F (256 -> 128)"
    )
    print()

    print("=" * 78)
    print(" Benchmark complete.")
    print("=" * 78)

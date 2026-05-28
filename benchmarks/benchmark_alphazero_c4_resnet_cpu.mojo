"""Benchmark AlphaZero ConnectFour Fused-ResNet PredModel on CPU.

Mirrors the structure of benchmark_alphazero_ttt_cnn_cpu.mojo, but for
`AlphaZeroConnectFourFusedResNetConfig.PredModel`:

    Sequential[
        Conv2DBatchNormReLU[3, F, 3, 1, 1, 6, 7],   # initial conv
        Repeat[N, ResBlockConv2DBN[F, 3, 1, 6, 7], shared=False],
        Parallel[
            # Policy head
            Sequential[
                Conv2DBatchNormReLU[F, HF, 1, 1, 0, 6, 7],
                FlattenLayer[HF*42],
                Linear[HF*42, 7],
            ],
            # Value head
            Sequential[
                Conv2DBatchNormReLU[F, HF, 1, 1, 0, 6, 7],
                FlattenLayer[HF*42],
                LinearReLU[HF*42, F],
                Linear[F, 1],
            ],
        ],
    ]

Defaults match the config: FILTERS=128, NUM_BLOCKS=5, HEAD_FILTERS=32.
Run:
    pixi run mojo run -I . benchmarks/benchmark_alphazero_c4_resnet_cpu.mojo
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import (
    Conv2DBatchNormReLU,
    Linear,
    LinearReLU,
    Model,
)
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN


@always_inline
def fill_random(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](random_float64(-1.0, 1.0))


@always_inline
def fmt_ms(ms: Float64) -> String:
    return String(ms)


@always_inline
def fmt_gflops(g: Float64) -> String:
    return String(g)


@fieldwise_init
struct LayerTime(Copyable, Movable):
    var fwd_ms: Float64
    var bwd_ms: Float64


def bench_layer[
    M: Model, BATCH: Int, FWD_ITERS: Int, BWD_ITERS: Int
](label: String) raises -> LayerTime:
    """Time forward + backward for any Model. Returns LayerTime."""
    comptime IN_DIM = M.IN_DIM
    comptime OUT_DIM = M.OUT_DIM
    comptime PARAM_SIZE = M.PARAM_SIZE
    comptime STATE_SIZE = M.STATE_SIZE
    comptime CACHE_SIZE = M.CACHE_SIZE

    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * max(1, CACHE_SIZE))

    fill_random(input_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    memset(output_buf, 0, BATCH * OUT_DIM)
    memset(grad_in_buf, 0, BATCH * IN_DIM)
    memset(grads_buf, 0, PARAM_SIZE)
    memset(cache_buf, 0, BATCH * max(1, CACHE_SIZE))
    memset(state_buf, 0, max(1, STATE_SIZE))

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    M.initialize_params[Kaiming[], dtype](params_lt)
    comptime if STATE_SIZE > 0:
        M.initialize_state[dtype](state_lt)

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
    var grads_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](grads_buf)

    # Warmup
    M.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    M.backward[BATCH, dtype](
        grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
    )

    var t0 = perf_counter_ns()
    for _ in range(FWD_ITERS):
        M.forward[BATCH, dtype](
            input_lt, output_lt, params_lt, state_lt, cache_lt
        )
    var t1 = perf_counter_ns()
    var fwd_ms = Float64(t1 - t0) / 1e6 / Float64(FWD_ITERS)

    var t2 = perf_counter_ns()
    for _ in range(BWD_ITERS):
        M.backward[BATCH, dtype](
            grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
        )
    var t3 = perf_counter_ns()
    var bwd_ms = Float64(t3 - t2) / 1e6 / Float64(BWD_ITERS)

    print("─" * 78)
    print(label)
    print(
        "  IN_DIM=",
        IN_DIM,
        " OUT_DIM=",
        OUT_DIM,
        " PS=",
        PARAM_SIZE,
        " CACHE/sample=",
        CACHE_SIZE,
        " BATCH=",
        BATCH,
    )
    print("  forward (train) :", fmt_ms(fwd_ms), "ms")
    print("  backward        :", fmt_ms(bwd_ms), "ms")

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()

    return LayerTime(fwd_ms, bwd_ms)


def main() raises:
    seed(42)

    comptime BATCH = 64
    comptime F = 128       # FILTERS
    comptime HF = 32       # HEAD_FILTERS
    comptime N_BLOCKS = 5  # ResBlock count
    comptime BOARD_H = 6
    comptime BOARD_W = 7
    comptime SPATIAL = BOARD_H * BOARD_W  # 42
    comptime HEAD_DIM = HF * SPATIAL      # 1344

    print("=" * 78)
    print(
        " AlphaZero ConnectFour Fused-ResNet PredModel — CPU layer benchmark"
    )
    print(
        "   FILTERS=",
        F,
        "  HEAD_FILTERS=",
        HF,
        "  N_BLOCKS=",
        N_BLOCKS,
        "  board=",
        BOARD_H,
        "x",
        BOARD_W,
        "  BATCH=",
        BATCH,
    )
    print("=" * 78)
    print()

    # ── Initial conv 3 → F, 3x3 same. ──────────────────────────────
    var init_t = bench_layer[
        Conv2DBatchNormReLU[3, F, 3, 1, 1, BOARD_H, BOARD_W], BATCH, 20, 10
    ](
        "Initial conv: Conv2DBatchNormReLU[3, F, 3x3 same, "
        + String(BOARD_H)
        + "x"
        + String(BOARD_W)
        + "]"
    )

    # ── One ResBlock body F → F, 3x3 same (×N_BLOCKS in PredModel). ─
    var block_t = bench_layer[
        ResBlockConv2DBN[F, 3, 1, BOARD_H, BOARD_W], BATCH, 10, 5
    ](
        "Body block: ResBlockConv2DBN[F, 3x3, "
        + String(BOARD_H)
        + "x"
        + String(BOARD_W)
        + "]  (×"
        + String(N_BLOCKS)
        + " blocks in PredModel)"
    )

    # ── Head conv F → HF, 1x1 valid (used in BOTH policy and value heads). ─
    var head_conv_t = bench_layer[
        Conv2DBatchNormReLU[F, HF, 1, 1, 0, BOARD_H, BOARD_W], BATCH, 30, 15
    ](
        "Head 1x1 conv: Conv2DBatchNormReLU[F, HF, 1x1, "
        + String(BOARD_H)
        + "x"
        + String(BOARD_W)
        + "]  (×2 — both heads)"
    )

    # ── Policy head FC: Linear[HEAD_DIM, 7]. ────────────────────────
    var pol_t = bench_layer[Linear[HEAD_DIM, 7], BATCH, 100, 50](
        "Policy head: Linear[" + String(HEAD_DIM) + ", 7]"
    )

    # ── Value head FC1 + FC2. ───────────────────────────────────────
    var val_fc1_t = bench_layer[LinearReLU[HEAD_DIM, F], BATCH, 100, 50](
        "Value head FC1: LinearReLU[" + String(HEAD_DIM) + ", F=" + String(F) + "]"
    )

    var val_fc2_t = bench_layer[Linear[F, 1], BATCH, 200, 100](
        "Value head FC2: Linear[F=" + String(F) + ", 1]"
    )

    # ── PredModel cost summary (one forward+backward training step). ─
    var init_fwd = init_t.fwd_ms
    var init_bwd = init_t.bwd_ms
    var block_fwd = block_t.fwd_ms
    var block_bwd = block_t.bwd_ms
    var head_conv_fwd = head_conv_t.fwd_ms
    var head_conv_bwd = head_conv_t.bwd_ms
    var pol_fwd = pol_t.fwd_ms
    var pol_bwd = pol_t.bwd_ms
    var val_fc1_fwd = val_fc1_t.fwd_ms
    var val_fc1_bwd = val_fc1_t.bwd_ms
    var val_fc2_fwd = val_fc2_t.fwd_ms
    var val_fc2_bwd = val_fc2_t.bwd_ms

    var blocks_fwd = Float64(N_BLOCKS) * block_fwd
    var blocks_bwd = Float64(N_BLOCKS) * block_bwd
    var heads_conv_fwd = 2.0 * head_conv_fwd
    var heads_conv_bwd = 2.0 * head_conv_bwd
    var total_fwd = (
        init_fwd
        + blocks_fwd
        + heads_conv_fwd
        + pol_fwd
        + val_fc1_fwd
        + val_fc2_fwd
    )
    var total_bwd = (
        init_bwd
        + blocks_bwd
        + heads_conv_bwd
        + pol_bwd
        + val_fc1_bwd
        + val_fc2_bwd
    )
    var total = total_fwd + total_bwd

    print()
    print("=" * 78)
    print(" Per-step summary (one training fwd+bwd at BATCH=", BATCH, ")")
    print("=" * 78)
    print(
        "  Initial conv               : ",
        fmt_ms(init_fwd),
        "+",
        fmt_ms(init_bwd),
        "=",
        fmt_ms(init_fwd + init_bwd),
        "ms",
    )
    print(
        "  ",
        N_BLOCKS,
        "× ResBlock body            : ",
        fmt_ms(blocks_fwd),
        "+",
        fmt_ms(blocks_bwd),
        "=",
        fmt_ms(blocks_fwd + blocks_bwd),
        "ms",
    )
    print(
        "  2 × 1x1 head conv          : ",
        fmt_ms(heads_conv_fwd),
        "+",
        fmt_ms(heads_conv_bwd),
        "=",
        fmt_ms(heads_conv_fwd + heads_conv_bwd),
        "ms",
    )
    print(
        "  Policy head Linear         : ",
        fmt_ms(pol_fwd),
        "+",
        fmt_ms(pol_bwd),
        "=",
        fmt_ms(pol_fwd + pol_bwd),
        "ms",
    )
    print(
        "  Value head LinearReLU+Linr : ",
        fmt_ms(val_fc1_fwd + val_fc2_fwd),
        "+",
        fmt_ms(val_fc1_bwd + val_fc2_bwd),
        "=",
        fmt_ms(val_fc1_fwd + val_fc2_fwd + val_fc1_bwd + val_fc2_bwd),
        "ms",
    )
    print("  " + "─" * 70)
    print(
        "  Total per step             : ",
        fmt_ms(total_fwd),
        "+",
        fmt_ms(total_bwd),
        "=",
        fmt_ms(total),
        "ms",
    )
    print("=" * 78)

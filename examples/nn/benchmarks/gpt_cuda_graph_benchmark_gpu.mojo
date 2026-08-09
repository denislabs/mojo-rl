"""GPT CUDA-graph throughput benchmark — capture vs eager (NVIDIA).

Times steady-state next-token training throughput of the `AutoregressiveTrainer`
with `USE_TRAIN_CUDA_GRAPH` OFF (eager: one `cuLaunchKernel` per primitive,
~hundreds/step) vs ON (the per-step device compute is captured once and replayed
as a single `cuGraphLaunch`/step). The data-build (host window sample + one-hot +
resident upload) and the cosine-LR push stay eager in BOTH modes, so this
isolates the kernel-launch-overhead win from the capture.

Each mode: build the GPT, run BENCH_ITERS as warmup (the capture mode captures
the graph on the first step), synchronize, then time a second BENCH_ITERS of
pure training (no eval). Steps/s + speedup are printed. The two models are built
SEQUENTIALLY (each dies before the next) so peak memory == one model.

⚠️ Capture is fp32-only (this GPT is fp32) and NVIDIA-only for real capture; on
non-NVIDIA `maybe_capture_replay` runs eagerly so both columns are the eager
path (speedup ≈ 1.0). On NVIDIA, whether capture wins depends on how
launch-bound the config is (smaller BATCH/seq = more launch-bound = bigger win).

Run on NVIDIA:
    pixi run -e nvidia mojo run -I . \
        examples/nn/benchmarks/gpt_cuda_graph_benchmark_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CharTokenizer, load_text, train_val_split
from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.gpt import GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie
from mojo_rl.nn.optimizer.adam import AdamW
from mojo_rl.nn.training.autoregressive_trainer import AutoregressiveTrainer
from mojo_rl.nn.core.initializer import Normal


# Same nanoGPT-class config as the training example.
comptime VOCAB = 65
comptime SEQ = 256
comptime EMBED = 384
comptime HEADS = 6
comptime LAYERS = 6
comptime FF_MULT = 4
comptime BATCH = 64

comptime BASE_LR: Scalar[DT] = 1e-3
comptime BETA2: Scalar[DT] = 0.99
comptime WD: Scalar[DT] = 0.1
comptime MIN_LR_SCALE: Float64 = 0.1
comptime WARMUP_ITERS = 100
comptime USE_MAX_ATTN = True
comptime DROPOUT_P: Float64 = 0.2
comptime GRAD_CLIP: Scalar[DT] = 1.0

# Per-phase iteration count (warmup phase + timed phase each run this many).
comptime BENCH_ITERS = 300

comptime GPT_MODEL = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
    UInt64(0xC0FFEE), USE_MAX_ATTN,
]


def bench[
    CAP: Bool
](ctx: DeviceContext, ref text: String) raises -> Float64:
    """Build the GPT once, warm up BENCH_ITERS (captures the graph when CAP),
    then time a second BENCH_ITERS of pure training. Returns steps/s. The
    trainer is destroyed at return so the next mode peaks at one model."""
    comptime AR = AutoregressiveTrainer[
        GPT_MODEL, AdamW, VOCAB, SEQ, BATCH, target="gpu",
        USE_TRAIN_CUDA_GRAPH=CAP,
    ]
    var tok = CharTokenizer(text)
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)

    var net = GPT_MODEL.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var optim = AdamW(lr=BASE_LR, beta2=BETA2, wd=WD)
    var artr = AR.make_from(
        net^, optim^, tok^, split^, ctx,
        BASE_LR, WARMUP_ITERS, BENCH_ITERS, MIN_LR_SCALE, GRAD_CLIP,
    )
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        UInt64(0xC0FFEE), USE_MAX_ATTN,
    ](artr.net, Optional(ctx))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        UInt64(0xC0FFEE), USE_MAX_ATTN,
    ](artr.net)
    ctx.synchronize()

    # Warmup phase: allocates module buffers + (CAP) captures the graph.
    _ = artr.fit(eval_every=0, n_val_windows=0, print_progress=False)
    ctx.synchronize()

    # Timed phase: pure training, no eval.
    var t0 = perf_counter_ns()
    _ = artr.fit(eval_every=0, n_val_windows=0, print_progress=False)
    ctx.synchronize()
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    return Float64(BENCH_ITERS) / elapsed


def main() raises:
    seed(42)
    print("=" * 70)
    print("GPT CUDA-graph throughput benchmark — capture vs eager")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS) + " batch=" + String(BATCH)
        + " | bench_iters=" + String(BENCH_ITERS) + " (×2 phases/mode)"
    )

    print("\n[data] loading TinyShakespeare...")
    var text = load_text()

    # ONE DeviceContext for both modes: the CUDA-graph interceptor records the
    # Mojo stream from the first kernel launch globally, so capture must run on
    # the SAME context/stream (two contexts → cuStreamBeginCapture fails 400).
    # The eager trainer is freed before the capture trainer is built (each in
    # its own `bench` scope), so peak memory is still one model.
    var ctx = DeviceContext()

    print("[bench] eager (USE_TRAIN_CUDA_GRAPH=False) ...")
    var eager_sps = bench[False](ctx, text)
    print("  eager:   " + String(eager_sps) + " steps/s")

    print("[bench] capture (USE_TRAIN_CUDA_GRAPH=True) ...")
    var cap_sps = bench[True](ctx, text)
    print("  capture: " + String(cap_sps) + " steps/s")

    print(
        "\n  speedup (capture / eager) = "
        + String(cap_sps / eager_sps) + "x"
    )
    print("=" * 70)

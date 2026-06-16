"""Trainer benchmark: AMP (fp32 vs bf16) across model sizes, + CUDA-graph hook.

Measures per-step train throughput of the nn `Trainer` and how the AMP
policy (`NoAMP` = all fp32 vs `Bf16Compute` = bf16 matmul compute, fp32
master weights / accumulators — `mojo_rl/nn/core/amp.mojo`) pays off as the
matmuls grow.

Two sweeps, all on one fixed MNIST mini-batch (loaded into the trainer's
device buffers once — isolates train-step cost from data movement):

  1. **MLP width sweep** — 784→H→H/2→10 for H in {256, 1024, 4096}. bf16 is a
     net LOSS on the smallest net (the per-layer fp32→bf16 / bf16→fp32 cast
     kernels cost more than the tiny GEMM saves) and should overtake fp32 as H
     grows and the GEMMs get tensor-core-bound. This sweep finds the crossover.

  2. **LeNet conv** — Conv(1→16,5,s2)→Conv(16→32,5,s2)→Flatten→Linear. nn
     `Conv2D` now has a GPU bf16 path (its forward `col@Wᵀ` and backward
     `goᵀ@col` GEMMs run in bf16; the dx gather + CPU path stay fp32), so AMP
     accelerates the whole conv stack — bf16 should win once the conv GEMMs are
     tensor-core-bound (more channels / larger spatial / batch).

Larger BATCH also enlarges the GEMM M dim (more AMP benefit); BATCH=512 here.

Run (NVIDIA CUDA):
    pixi run -e nvidia mojo run -I . \
        examples/nn/benchmarks/trainer_graph_amp_benchmark_gpu.mojo
Run (Apple Metal — note: Metal has no bf16 tensor cores, so bf16 looks slow):
    pixi run -e apple mojo run -I . \
        examples/nn/benchmarks/trainer_graph_amp_benchmark_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import Module, AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay


comptime IN_DIM = 784
comptime N_CLASSES = 10
comptime BATCH = 512
comptime LR: Scalar[DT] = 0.001

comptime WARMUP = 20
comptime N_STEPS = 200

# CUDA-graph capture of the full train step is gated OFF by default: Modular's
# `linalg.matmul` GPU path allocates a split-K reduction workspace DeviceBuffer
# per call (freed at return), which is illegal during CUDA stream capture and
# aborts (`AsyncRT_DeviceBuffer_release`). The eager AMP columns are unaffected
# and run on any GPU. See the notes at the end of the file to enable.
comptime USE_CUDA_GRAPH = False

# ── Networks (all flat IN_DIM=784 → N_CLASSES=10; Conv2D reshapes internally) ──
comptime MLP_S = Sequential[
    Linear[IN_DIM, 256], ReLU[256],
    Linear[256, 128], ReLU[128],
    Linear[128, N_CLASSES],
]
comptime MLP_M = Sequential[
    Linear[IN_DIM, 1024], ReLU[1024],
    Linear[1024, 512], ReLU[512],
    Linear[512, N_CLASSES],
]
comptime MLP_L = Sequential[
    Linear[IN_DIM, 4096], ReLU[4096],
    Linear[4096, 1024], ReLU[1024],
    Linear[1024, N_CLASSES],
]
comptime CONV = Sequential[
    Conv2D[1, 16, 5, 2, 0, 28, 28], ReLU[16 * 12 * 12],
    Conv2D[16, 32, 5, 2, 0, 12, 12], ReLU[32 * 4 * 4],
    Flatten[32 * 4 * 4],
    Linear[32 * 4 * 4, N_CLASSES],
]


def bench_net[
    NET: Module, POLICY: AMPPolicy
](
    ctx: DeviceContext,
    batch_x: List[Scalar[DT]],
    batch_y: List[Scalar[DT]],
) raises -> Float64:
    """Build a trainer (given net + AMP policy), load the fixed batch, time
    N_STEPS eager device steps, and return elapsed seconds. When
    USE_CUDA_GRAPH is on, also captures+replays the step and prints a graph
    line (see top-of-file note on why it's off by default)."""
    var trainer = Trainer[
        NET,
        Adam,
        CrossEntropyLoss[N_CLASSES],
        BATCH,
        target="gpu",
        POLICY=POLICY,
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = LR
    trainer.load_fixed_batch(batch_x, batch_y)

    for _ in range(WARMUP):
        trainer.train_step_device()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        trainer.train_step_device()
    ctx.synchronize()
    var eager_s = Float64(perf_counter_ns() - t0) / 1e9

    comptime if USE_CUDA_GRAPH:
        var graph: Optional[CUDAGraph] = None

        def _step() capturing raises -> None:
            trainer.train_step_device()

        for _ in range(WARMUP):
            maybe_capture_replay[_step](graph, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for _ in range(N_STEPS):
            maybe_capture_replay[_step](graph, ctx)
        ctx.synchronize()
        var graph_s = Float64(perf_counter_ns() - t1) / 1e9
        print(
            "    graph "
            + String(graph_s)
            + "s ("
            + String(Float64(N_STEPS) / graph_s)
            + " steps/s)  graph-speedup "
            + String(eager_s / graph_s)
            + "x"
        )
    return eager_s


def bench_pair[
    NET: Module
](
    ctx: DeviceContext,
    batch_x: List[Scalar[DT]],
    batch_y: List[Scalar[DT]],
    tag: String,
) raises:
    """Time fp32 (NoAMP) vs bf16 (Bf16Compute) for one net and print a row
    with the bf16 speedup (>1 means bf16 is faster)."""
    var fp32_s = bench_net[NET, NoAMP](ctx, batch_x, batch_y)
    var bf16_s = bench_net[NET, Bf16Compute](ctx, batch_x, batch_y)
    var fp32_sps = Float64(N_STEPS) / fp32_s
    var bf16_sps = Float64(N_STEPS) / bf16_s
    print(
        tag
        + "  | fp32 "
        + String(fp32_sps)
        + " steps/s  | bf16 "
        + String(bf16_sps)
        + " steps/s  | bf16 speedup "
        + String(fp32_s / bf16_s)
        + "x"
    )


def main() raises:
    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    # One fixed BATCH of MNIST as flat host Lists, reused by every net
    # (all share IN_DIM=784, N_CLASSES=10).
    var batch_x = List[Scalar[DT]](length=BATCH * IN_DIM, fill=Scalar[DT](0.0))
    var batch_y = List[Scalar[DT]](
        length=BATCH * N_CLASSES, fill=Scalar[DT](0.0)
    )
    for i in range(BATCH * IN_DIM):
        batch_x[i] = ds.train_images[i]
    for i in range(BATCH):
        batch_y[i * N_CLASSES + Int(ds.train_labels[i])] = Scalar[DT](1.0)

    print(
        "config: BATCH="
        + String(BATCH)
        + "  WARMUP="
        + String(WARMUP)
        + "  N_STEPS="
        + String(N_STEPS)
        + "  cuda_graph="
        + String(USE_CUDA_GRAPH)
    )
    print("(bf16 speedup > 1 = bf16 faster; Apple Metal has no bf16 tensor cores)\n")

    print("== MLP width sweep (where AMP crosses over) ==")
    bench_pair[MLP_S](ctx, batch_x, batch_y, "mlp 784-256-128-10  ")
    bench_pair[MLP_M](ctx, batch_x, batch_y, "mlp 784-1024-512-10 ")
    bench_pair[MLP_L](ctx, batch_x, batch_y, "mlp 784-4096-1024-10")

    print("\n== LeNet conv (Conv2D has a GPU bf16 path; dx gather stays fp32) ==")
    bench_pair[CONV](ctx, batch_x, batch_y, "conv lenet          ")

    print("\nDONE")


# ──────────────────────────────────────────────────────────────────────────
# Notes.
#
# AMP (Bf16Compute): `Linear` and `Conv2D` both have a GPU bf16 compute path —
# cast fp32 weights/inputs → bf16, run the bf16 GEMM, cast the output back to
# fp32 (fwd + bwd). Those casts are fixed per-call overhead, so bf16 only wins
# once the GEMM is large enough (big hidden width / channels / spatial / batch).
# Conv2D's dx step is a gather kernel (not a GEMM) and its CPU path stay fp32.
#
# CUDA-graph capture (USE_CUDA_GRAPH): the Trainer step is capturable (Adam
# keeps its step counter/bias-correction on-device, `forward_capture` drops the
# loss host readback, fixed device buffers). The blocker is the GEMM:
# `linalg.matmul` picks a split-K path for some shapes and allocs+frees a
# workspace DeviceBuffer per call — a cudaFree inside the captured stream,
# which aborts. To enable: make the capture-path matmul allocation-free
# (force num_k_partitions=1, or route Linear/Conv2D through the alloc-free
# in-repo `mojo_rl/nn/gpu/matmul.mojo::gpu_matmul`), then set USE_CUDA_GRAPH=True.
# ──────────────────────────────────────────────────────────────────────────

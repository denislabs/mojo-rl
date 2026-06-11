"""Trainer benchmark: CUDA-graph capture × AMP (fp32 vs bf16) on a GPU step.

Measures the per-step throughput of the nn2 `Trainer` on an MLP MNIST step
under four configurations:

    fp32  eager   |  fp32  graph
    bf16  eager   |  bf16  graph

  - **eager**: each step enqueues its kernels on the Mojo stream directly.
  - **graph**: the first step is captured into a CUDA graph via
    `maybe_capture_replay`; every later step is a single graph replay
    (no per-kernel launch overhead).
  - **fp32 / bf16**: the `Trainer`'s `POLICY` comptime param — `NoAMP`
    (all fp32) vs `Bf16Compute` (bf16 matmul compute, fp32 master weights /
    accumulators). See `mojo_rl/nn2/core/amp.mojo`.

The benchmark trains repeatedly on ONE fixed mini-batch (loaded into the
trainer's device buffers once). That isolates the train-step cost — exactly
what CUDA-graph capture targets (kernel-launch overhead) — from data
movement / shuffling.

Platform note: real CUDA-graph capture/replay requires NVIDIA. On Apple /
non-NVIDIA, `maybe_capture_replay` is a compile-time no-op and the "graph"
columns run eagerly (so eager ≈ graph there) — the example still compiles
and runs as a correctness smoke for the capturable code path.

Run (NVIDIA CUDA):
    pixi run -e nvidia mojo run -I . \
        examples/nn2/benchmarks/trainer_graph_amp_benchmark_gpu.mojo
Run (Apple Metal — eager-only smoke):
    pixi run -e apple mojo run -I . \
        examples/nn2/benchmarks/trainer_graph_amp_benchmark_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay


comptime IN_DIM = 784
comptime H1 = 256
comptime H2 = 128
comptime N_CLASSES = 10
comptime BATCH = 256
comptime LR: Scalar[DT] = 0.001

comptime WARMUP = 20
comptime N_STEPS = 200

comptime Net = Sequential[
    Linear[IN_DIM, H1],
    ReLU[H1],
    Linear[H1, H2],
    ReLU[H2],
    Linear[H2, N_CLASSES],
]


def run_policy[
    POLICY: AMPPolicy
](
    ctx: DeviceContext,
    batch_x: List[Scalar[DT]],
    batch_y: List[Scalar[DT]],
    label: String,
) raises:
    """Build a trainer with the given AMP policy, load the fixed batch, then
    time `N_STEPS` train steps eagerly and via CUDA-graph replay."""
    var trainer = Trainer[
        Net,
        Adam,
        CrossEntropyLoss[N_CLASSES],
        BATCH,
        target="gpu",
        POLICY=POLICY,
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = LR
    trainer.load_fixed_batch(batch_x, batch_y)

    # ---- eager ----
    for _ in range(WARMUP):
        trainer.train_step_device()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        trainer.train_step_device()
    ctx.synchronize()
    var eager_s = Float64(perf_counter_ns() - t0) / 1e9

    # ---- graph ----
    # `_step` is the captured body — one pure-device train step. The first
    # `maybe_capture_replay` call captures it; the rest replay.
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

    var eager_sps = Float64(N_STEPS) / eager_s
    var graph_sps = Float64(N_STEPS) / graph_s
    print(
        label
        + "  | eager "
        + String(eager_s)
        + "s ("
        + String(eager_sps)
        + " steps/s)  | graph "
        + String(graph_s)
        + "s ("
        + String(graph_sps)
        + " steps/s)  | graph speedup "
        + String(eager_s / graph_s)
        + "x"
    )


def main() raises:
    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    # Extract the first BATCH examples as flat host Lists: x = [BATCH, IN_DIM],
    # y = one-hot [BATCH, N_CLASSES].
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
        + " net=784-256-128-10  WARMUP="
        + String(WARMUP)
        + " N_STEPS="
        + String(N_STEPS)
    )
    print("(Apple/non-NVIDIA: graph columns run eagerly — no-op capture)\n")

    run_policy[NoAMP](ctx, batch_x, batch_y, "fp32")
    run_policy[Bf16Compute](ctx, batch_x, batch_y, "bf16")
    print("\nDONE")

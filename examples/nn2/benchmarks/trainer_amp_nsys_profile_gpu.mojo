"""Focused nsys profiling harness: fp32 vs bf16 AMP, ONE net per invocation.

The width-sweep benchmark (`trainer_graph_amp_benchmark_gpu.mojo`) showed bf16
plateauing at ~0.98× on NVIDIA even for the fat 784→4096→1024 MLP — where bf16
tensor cores *should* win clearly. To find out where the time actually goes
(tensor-core GEMM vs the fp32→bf16 / bf16→fp32 cast kernels vs the fp32 Adam
step vs activations), profile each policy in ISOLATION under nsys.

Why one policy per process: nsys aggregates by kernel name. fp32 and bf16 share
the matmul / Adam / ReLU kernels, so running both in one process sums them and
destroys attribution. Running one policy per invocation keeps each trace pure —
the cast kernels (`_fp32_to_bf16_kernel` / `_bf16_to_fp32_kernel`) appear ONLY
in the bf16 trace, and the GEMM time is directly comparable across the two
traces.

Usage (NVIDIA):
    # fp32 (NoAMP) trace
    pixi run -e nvidia nsys profile --stats=true -o /tmp/amp_fp32 \
        mojo run -I . examples/nn2/benchmarks/trainer_amp_nsys_profile_gpu.mojo fp32

    # bf16 (Bf16Compute) trace
    pixi run -e nvidia nsys profile --stats=true -o /tmp/amp_bf16 \
        mojo run -I . examples/nn2/benchmarks/trainer_amp_nsys_profile_gpu.mojo bf16

    # optional net selector (default = the fat MLP, the red flag): mlp | conv
    ... trainer_amp_nsys_profile_gpu.mojo bf16 conv

Then read the per-kernel breakdown:
    nsys stats --report gpukernsum /tmp/amp_fp32.nsys-rep
    nsys stats --report gpukernsum /tmp/amp_bf16.nsys-rep

What to look at in `gpukernsum` (Total Time % per kernel):
  - bf16 GEMM total time vs fp32 GEMM total time  → is the GEMM actually faster?
    (if not, `linalg.matmul` may not be hitting tensor cores for these shapes)
  - `_fp32_to_bf16_kernel` + `_bf16_to_fp32_kernel` total %  → cast overhead
  - Adam / elementwise / ReLU / CE %  → the fp32 floor AMP can't touch (Amdahl)

Decision rule:
  - casts dominate  → cast weight once per step (share fwd+bwd), expect a real win
  - fp32 floor dominates  → bf16 can't win at MLP sizes; reserve AMP for conv/heavy GEMM
  - GEMM not faster in bf16  → tensor cores aren't engaged; that's the real bug

Note: WARMUP steps are included in the trace. They run the same kernels as the
measured steps, so per-kernel *time share* is unaffected; only absolute counts
include the warmup. Keep WARMUP << N_STEPS.
"""

from std.sys import argv
from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Module, AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


comptime IN_DIM = 784
comptime N_CLASSES = 10
comptime BATCH = 512
comptime LR: Scalar[DT] = 0.001

comptime WARMUP = 20
comptime N_STEPS = 500  # more steps than the sweep — a denser, cleaner trace

# The fat MLP is the red flag (bf16 should win, doesn't). Conv is the secondary.
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


def run_net[
    NET: Module, POLICY: AMPPolicy
](
    ctx: DeviceContext,
    batch_x: List[Scalar[DT]],
    batch_y: List[Scalar[DT]],
    tag: String,
) raises:
    """Build trainer (net + policy), load the fixed batch, warm up, then time
    N_STEPS device steps. The whole region runs under nsys — read the kernel
    breakdown afterward with `nsys stats --report gpukernsum`."""
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
    var elapsed_s = Float64(perf_counter_ns() - t0) / 1e9

    print(
        tag
        + "  | "
        + String(N_STEPS)
        + " steps in "
        + String(elapsed_s)
        + "s  | "
        + String(Float64(N_STEPS) / elapsed_s)
        + " steps/s"
    )


def main() raises:
    seed(42)

    # argv: [prog, policy, net?]   policy ∈ {fp32, bf16}   net ∈ {mlp, conv}
    if len(argv()) < 2:
        print(
            "usage: ... trainer_amp_nsys_profile_gpu.mojo <fp32|bf16> [mlp|conv]"
        )
        return
    var policy = String(argv()[1])
    var net = String("mlp")
    if len(argv()) > 2:
        net = String(argv()[2])

    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

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
        + "  policy="
        + policy
        + "  net="
        + net
    )

    var tag = "[" + net + " / " + policy + "]"
    if net == "conv":
        if policy == "bf16":
            run_net[CONV, Bf16Compute](ctx, batch_x, batch_y, tag)
        else:
            run_net[CONV, NoAMP](ctx, batch_x, batch_y, tag)
    else:
        if policy == "bf16":
            run_net[MLP_L, Bf16Compute](ctx, batch_x, batch_y, tag)
        else:
            run_net[MLP_L, NoAMP](ctx, batch_x, batch_y, tag)

    print("DONE")

"""BF16-flow (AMP) supervised Trainer smoke (GPU).

Builds a bf16 model `Sequential[Linear[784,128,bf16], Linear[128,10,bf16]]` —
its `ACT_DT` is `bfloat16`, so the Trainer's `MADT` is bf16 and the AMP boundary
casts (fp32 dataset → bf16 input, bf16 logits → fp32 loss, fp32 grad → bf16) all
fire. This is a WIRING smoke: it confirms the bf16-flow Trainer COMPILES and RUNS
on the GPU without crashing. Numerics on Apple Metal's bf16 are not meaningful
(Metal's bf16 is broken) — real accuracy is an NVIDIA gate. Uses a tiny synthetic
classification set (no MNIST download) so it runs fast.

Run (Apple):  pixi run -e apple mojo run -I . examples/nn/mlp/mlp_bf16_amp_trainer_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/mlp/mlp_bf16_amp_trainer_gpu.mojo
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.training.trainer import Trainer


def main() raises:
    comptime assert has_accelerator(), "bf16 AMP trainer smoke requires a GPU"
    comptime IN_DIM = 784
    comptime H1 = 128
    comptime NC = 10
    comptime BATCH = 50
    comptime N_TRAIN = 200  # 4 batches
    comptime N_TEST = 100  # 2 batches

    var c = DeviceContext()

    # bf16-flow model: both Linears flow activations at bf16 → Net.ACT_DT == bf16.
    comptime Net = Sequential[
        Linear[IN_DIM, H1, DType.bfloat16],
        Linear[H1, NC, DType.bfloat16],
    ]
    comptime assert (
        Net.ACT_DT == DType.bfloat16
    ), "bf16 net must derive ACT_DT == bfloat16"

    print("constructing bf16-flow Trainer (MADT = bfloat16)...")
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-3
    )

    # Synthetic data: train_x = [N_TRAIN*IN] in [0,1), one-hot labels cycling
    # over classes; test mirrors it.
    var train_x = List[Scalar[DT]](length=N_TRAIN * IN_DIM, fill=0.0)
    var train_y = List[Scalar[DT]](length=N_TRAIN * NC, fill=0.0)
    for i in range(N_TRAIN):
        var cls = i % NC
        train_y[i * NC + cls] = 1.0
        for j in range(IN_DIM):
            train_x[i * IN_DIM + j] = Scalar[DT](Float64((i * 31 + j) % 97) / 97.0)
    var test_x = List[Scalar[DT]](length=N_TEST * IN_DIM, fill=0.0)
    var test_labels = List[Int32](length=N_TEST, fill=0)
    for i in range(N_TEST):
        test_labels[i] = Int32(i % NC)
        for j in range(IN_DIM):
            test_x[i * IN_DIM + j] = Scalar[DT](Float64((i * 17 + j) % 89) / 89.0)

    print("running a few bf16 train_epoch + eval_top1 steps...")
    for ep in range(3):
        var loss = trainer.train_epoch[N_TRAIN](train_x, train_y, Optional(c))
        var acc = trainer.eval_top1[N_TEST](test_x, test_labels, Optional(c))
        print(
            "epoch " + String(ep) + " | train_loss=" + String(loss)
            + " | test_top1=" + String(acc * 100.0) + "%"
        )

    print("running a few bf16 train_gpu steps (shuffle)...")
    var result = trainer.train_gpu[N_TRAIN, N_TEST](
        train_x,
        train_y,
        test_x,
        test_labels,
        Optional(c),
        epochs=2,
        shuffle=True,
    )
    print("train_gpu epochs run: " + String(len(result.epoch_train_loss)))
    print("bf16 AMP Trainer wiring COMPILES + RUNS — DONE")

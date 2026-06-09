"""Regression: multi-conv GPU training must converge on VARIED data.

Guards the Conv2D GPU backward-order invariant. When a conv is not the
first layer, Sequential aliases its grad_input slab onto the buffer
holding its cached forward input. The GPU queue is in-order, so the
param-grad kernels (which read the input) MUST be enqueued before the
dx kernel (which overwrites that buffer). If they aren't, the second
conv's dW is computed from gradients instead of activations — silent on
single-step parity and on single-batch overfit (the stale input equals
the current one), but it corrupts multi-batch training: the net learns
for ~1 epoch then diverges back to uniform (loss → log(10) ≈ 2.302).

This test trains a 2-conv LeNet on 40 distinct MNIST batches for a few
epochs and asserts the loss drops well below the random baseline.
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


def main() raises:
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime IN_DIM = 1 * 28 * 28
    comptime N_BATCHES = 40
    comptime EPOCHS = 4
    comptime LR: Scalar[DT] = 0.001

    seed(42)
    print("test_conv_multiconv_train_gpu ...")
    var ds = MNIST()
    var ctx = DeviceContext()

    comptime Net = Sequential[
        Conv2D[1, 16, 5, 2, 0, 28, 28], ReLU[16 * 12 * 12],
        Conv2D[16, 32, 5, 2, 0, 12, 12], ReLU[32 * 4 * 4],
        Flatten[32 * 4 * 4],
        Linear[32 * 4 * 4, N_CLASSES],
    ]
    var trainer = Trainer[
        Net, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu",
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = LR

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N_CLASSES)

    var last_loss: Scalar[DT] = 0.0
    for epoch in range(EPOCHS):
        var ep_loss: Scalar[DT] = 0.0
        for bi in range(N_BATCHES):
            for b in range(BATCH):
                var idx = bi * BATCH + b
                for px in range(IN_DIM):
                    in_buf[b * IN_DIM + px] = ds.train_images[idx * IN_DIM + px]
                for c in range(N_CLASSES):
                    tgt_buf[b * N_CLASSES + c] = 0.0
                tgt_buf[b * N_CLASSES + Int(ds.train_labels[idx])] = 1.0
            ep_loss += trainer.train_step(in_buf, tgt_buf)
        last_loss = ep_loss / Scalar[DT](N_BATCHES)
        print("  epoch", epoch, " avg_loss=", last_loss)
    in_buf.free(); tgt_buf.free()

    assert_true(
        last_loss < Scalar[DT](0.6),
        "multi-conv GPU training diverged (loss "
        + String(last_loss)
        + " ≥ 0.6) — Conv2D backward order regression?",
    )
    print("  ok")

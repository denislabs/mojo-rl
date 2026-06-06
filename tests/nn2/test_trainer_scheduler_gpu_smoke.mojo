"""Trainer LR-scheduler smoke (GPU): MLP on an MNIST subset via train_gpu
with WarmupCosineSchedule. Validates the SCHEDULER hook compiles + runs,
the net still learns, and the optimizer LR is actually scaled (decayed
below the base LR by the final epoch).
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer, WarmupCosineSchedule
from mojo_rl.nn2.initializer import Kaiming


def main() raises:
    comptime IN_DIM = 784
    comptime N_CLASSES = 10
    comptime BATCH = 100
    comptime NTR = 2000
    comptime NTE = 1000
    comptime EPOCHS = 5
    comptime BASE_LR: Scalar[DT] = 0.001

    seed(42)
    print("test_trainer_scheduler_gpu_smoke ...")
    var ds = MNIST()
    var ctx = DeviceContext()

    comptime Net = Sequential[
        Linear[IN_DIM, 128], ReLU[128], Linear[128, N_CLASSES],
    ]
    var trainer = Trainer[
        Net, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu",
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = BASE_LR

    var xh = ctx.enqueue_create_host_buffer[DT](NTR * IN_DIM)
    var yh = ctx.enqueue_create_host_buffer[DT](NTR * N_CLASSES)
    var txh = ctx.enqueue_create_host_buffer[DT](NTE * IN_DIM)
    ctx.synchronize()
    for i in range(NTR * IN_DIM):
        xh.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(NTR * N_CLASSES):
        yh.unsafe_ptr()[i] = 0.0
    for i in range(NTR):
        yh.unsafe_ptr()[i * N_CLASSES + Int(ds.train_labels[i])] = 1.0
    for i in range(NTE * IN_DIM):
        txh.unsafe_ptr()[i] = ds.test_images[i]
    var tlbl: UnsafePointer[Int32, MutAnyOrigin] = (
        ctx.enqueue_create_host_buffer[DType.int32](NTE).unsafe_ptr()
    )
    for i in range(NTE):
        tlbl[i] = Int32(ds.test_labels[i])

    var xd = ctx.enqueue_create_buffer[DT](NTR * IN_DIM)
    var yd = ctx.enqueue_create_buffer[DT](NTR * N_CLASSES)
    var txd = ctx.enqueue_create_buffer[DT](NTE * IN_DIM)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(yd, yh)
    ctx.enqueue_copy(txd, txh)
    ctx.synchronize()

    var train_x = TileTensor(xd, row_major[NTR, IN_DIM]())
    var train_y = TileTensor(yd, row_major[NTR, N_CLASSES]())
    var test_x = TileTensor(txd, row_major[NTE, IN_DIM]())

    var result = trainer.train_gpu[
        NTR, NTE, SCHEDULER = WarmupCosineSchedule[2, 0.1]
    ](
        train_x, train_y, test_x, tlbl,
        epochs=EPOCHS, shuffle=True, rng_seed=UInt64(42),
    )

    var best: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best:
            best = a
    var final_lr = trainer.optim.get_lr()
    print("  best_acc =", best * 100.0, "%  base_lr =", BASE_LR, "  final_lr =", final_lr)
    assert_true(best > 0.5, "scheduled MLP should learn (>50% on 2k subset)")
    assert_true(
        final_lr < BASE_LR and final_lr > Scalar[DT](0.0),
        "WarmupCosine should decay LR below base by the last epoch",
    )
    print("  ok")

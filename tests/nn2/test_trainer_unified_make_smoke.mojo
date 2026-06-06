"""Trainer unified-make smoke (audit L2) — dataset-free.

`Trainer.make` was the last dual-make holdout (a CPU `make[INIT]()` +
a GPU `make[INIT](ctx)` overload). L2 collapses them into the nn2
convention `make[INIT](ctx: Optional[DeviceContext] = None)`. This smoke
exercises BOTH call forms on synthetic, separable data (no MNIST
dependency):

  * CPU:  `TRAINER_CPU.make[Kaiming]()`        (ctx defaults to None)
  * GPU:  `TRAINER_GPU.make[Kaiming](ctx)`     (DeviceContext → Optional)

and asserts a handful of train_steps drive the loss down on both, plus
`predict` runs. Run under `-e apple` so both targets build:

    pixi run -e apple mojo run -I . tests/nn2/test_trainer_unified_make_smoke.mojo
"""

from std.memory import alloc
from std.random import seed
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


comptime IN_DIM = 8
comptime H = 16
comptime N_CLASSES = 2
comptime BATCH = 16
comptime N_STEPS = 60

comptime MLP = Sequential[
    Linear[IN_DIM, H], ReLU[H], Linear[H, N_CLASSES],
]
comptime TRAINER_CPU = Trainer[
    MLP, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="cpu"
]
comptime TRAINER_GPU = Trainer[
    MLP, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu"
]


def _fill_separable(
    inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    # Class = (x[0] > 0); x[0] is the only informative feature.
    for b in range(BATCH):
        var sign: Scalar[DT] = 1.0 if (b % 2 == 0) else -1.0
        for d in range(IN_DIM):
            inp[b * IN_DIM + d] = Scalar[DT](0.1 * Float64(d))
        inp[b * IN_DIM] = sign
        var cls = 0 if sign > 0 else 1
        for c in range(N_CLASSES):
            tgt[b * N_CLASSES + c] = 0.0
        tgt[b * N_CLASSES + cls] = 1.0


def test_cpu() raises:
    print("--- CPU: TRAINER_CPU.make[Kaiming]() ---")
    seed(11)
    var trainer = TRAINER_CPU.make[Kaiming]()
    var inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_DIM
    )
    var tgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )
    _fill_separable(inp, tgt)
    var first = trainer.train_step(inp, tgt)
    var last: Scalar[DT] = first
    for _ in range(N_STEPS):
        last = trainer.train_step(inp, tgt)
    trainer.predict(inp, out)
    print("  loss", first, "->", last)
    assert_true(last < first, "CPU trainer should reduce the loss")
    inp.free(); tgt.free(); out.free()
    print("  ok")


def test_gpu(ctx: DeviceContext) raises:
    print("--- GPU: TRAINER_GPU.make[Kaiming](ctx) ---")
    seed(11)
    var trainer = TRAINER_GPU.make[Kaiming](ctx)
    var inp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_DIM
    )
    var tgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * N_CLASSES
    )
    _fill_separable(inp, tgt)
    var first = trainer.train_step(inp, tgt)
    var last: Scalar[DT] = first
    for _ in range(N_STEPS):
        last = trainer.train_step(inp, tgt)
    trainer.predict(inp, out)
    print("  loss", first, "->", last)
    assert_true(last < first, "GPU trainer should reduce the loss")
    inp.free(); tgt.free(); out.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Trainer unified-make smoke (L2)")
    print("=" * 70)
    test_cpu()
    var ctx = DeviceContext()
    test_gpu(ctx)
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

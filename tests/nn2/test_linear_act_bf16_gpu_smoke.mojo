"""Smoke: LinearAct (LinearReLU) GPU bf16 path compiles + runs finite.

Validates the AMP port into LinearAct — the three GEMMs (forward, grad_w,
grad_input) now run in bf16 under Bf16Compute, mirroring Linear. Builds a
fused-layer MLP, trains a few steps on a fixed batch, asserts the loss
stays finite (no NaN from the bf16 grad_w / weight-reuse paths).
"""

from std.random import seed
from std.math import isfinite
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import Bf16Compute


comptime IN_DIM = 64
comptime H = 128
comptime N_CLASSES = 10
comptime BATCH = 32

comptime NET = Sequential[
    LinearReLU[IN_DIM, H],
    LinearReLU[H, H],
    Linear[H, N_CLASSES],
]


def test_linear_act_bf16_gpu_smoke() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = Trainer[
        NET,
        Adam,
        CrossEntropyLoss[N_CLASSES],
        BATCH,
        target="gpu",
        POLICY=Bf16Compute,
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = 0.001

    var bx = List[Scalar[DT]](length=BATCH * IN_DIM, fill=Scalar[DT](0.1))
    var by = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=Scalar[DT](0.0))
    for i in range(BATCH):
        by[i * N_CLASSES + (i % N_CLASSES)] = Scalar[DT](1.0)
    var loss = Scalar[DT](0.0)
    for _ in range(10):
        loss = trainer.train_step(bx, by)
    ctx.synchronize()

    print("LinearAct bf16 GPU smoke — final loss:", loss)
    assert_true(
        isfinite(Float64(loss)), "LinearAct bf16 loss must be finite"
    )
    print("  [PASS] LinearAct bf16 GPU path runs finite")


def main() raises:
    test_linear_act_bf16_gpu_smoke()

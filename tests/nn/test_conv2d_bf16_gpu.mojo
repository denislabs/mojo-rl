"""Conv2D bf16 (AMP) GPU path — correctness smoke.

Validates the cast-around-matmul bf16 path added to `Conv2D` (forward
`col@Wᵀ` and backward `goᵀ@col` GEMMs run in bf16; the dx gather + CPU
path stay fp32). Trains a LeNet-style conv net on one fixed MNIST batch
under both `NoAMP` (fp32) and `Bf16Compute`, and asserts:

  - both reduce the loss and stay finite (the bf16 casts are wired right —
    a bad cast would NaN or fail to learn), and
  - with identical seed/init/batch, the bf16 run lands close to the fp32
    run (bf16 GEMM rounding only — not a different trajectory).

On Apple Metal bf16 runs (no tensor cores, but the kernels compile/run);
on NVIDIA it exercises the tensor-core bf16 GEMM.

Run:  pixi run -e apple  mojo run -I . tests/nn/test_conv2d_bf16_gpu.mojo
      pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_bf16_gpu.mojo
"""

from std.math import exp, log, isfinite, abs
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import MNIST
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.loss import CrossEntropyLoss
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer import Kaiming


comptime IN_DIM = 784
comptime N_CLASSES = 10
comptime BATCH = 64
comptime N_STEPS = 120

comptime Net = Sequential[
    Conv2D[1, 16, 5, 2, 0, 28, 28], ReLU[16 * 12 * 12],
    Conv2D[16, 32, 5, 2, 0, 12, 12], ReLU[32 * 4 * 4],
    Flatten[32 * 4 * 4],
    Linear[32 * 4 * 4, N_CLASSES],
]


def _ce_loss(
    logits: List[Scalar[DT]], targets_onehot: List[Scalar[DT]]
) -> Float64:
    var total: Float64 = 0.0
    for b in range(BATCH):
        var m = logits[b * N_CLASSES + 0]
        for c in range(1, N_CLASSES):
            if logits[b * N_CLASSES + c] > m:
                m = logits[b * N_CLASSES + c]
        var s: Float64 = 0.0
        for c in range(N_CLASSES):
            s += Float64(exp(logits[b * N_CLASSES + c] - m))
        var lse = Float64(m) + log(s)
        for c in range(N_CLASSES):
            var t = Float64(targets_onehot[b * N_CLASSES + c])
            total += -t * (Float64(logits[b * N_CLASSES + c]) - lse)
    return total / Float64(BATCH)


def _run[
    POLICY: AMPPolicy
](
    ctx: DeviceContext,
    x: List[Scalar[DT]],
    y: List[Scalar[DT]],
    mut loss_init: Float64,
) raises -> Float64:
    """Fresh seeded conv trainer; return final loss after N_STEPS, and report
    the (policy-independent) initial loss through `loss_init`."""
    seed(7)
    var trainer = Trainer[
        Net, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu", POLICY=POLICY
    ].make[INIT=Kaiming](ctx)
    trainer.optim.lr = Scalar[DT](0.001)
    trainer.load_fixed_batch(x, y)

    var out = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=Scalar[DT](0.0))
    trainer.predict(x, out)
    loss_init = _ce_loss(out, y)

    for _ in range(N_STEPS):
        trainer.train_step_device()
    ctx.synchronize()
    trainer.predict(x, out)
    return _ce_loss(out, y)


def test_conv2d_bf16_gpu() raises:
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    var x = List[Scalar[DT]](length=BATCH * IN_DIM, fill=Scalar[DT](0.0))
    var y = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=Scalar[DT](0.0))
    for i in range(BATCH * IN_DIM):
        x[i] = ds.train_images[i]
    for i in range(BATCH):
        y[i * N_CLASSES + Int(ds.train_labels[i])] = Scalar[DT](1.0)

    var init32: Float64 = 0.0
    var loss_fp32 = _run[NoAMP](ctx, x, y, init32)
    var init16: Float64 = 0.0
    var loss_bf16 = _run[Bf16Compute](ctx, x, y, init16)

    print("initial loss:", init32)
    print("fp32 final:", loss_fp32, " bf16 final:", loss_bf16)

    assert_true(isfinite(loss_fp32), "fp32 conv loss must be finite")
    assert_true(isfinite(loss_bf16), "bf16 conv loss must be finite")
    assert_true(
        loss_fp32 < init32 - 0.1, "fp32 conv must reduce the loss"
    )
    assert_true(
        loss_bf16 < init32 - 0.1,
        "bf16 conv must reduce the loss (cast-around-matmul wired right)",
    )
    # bf16 GEMM rounding only — same seed/init/batch ⇒ close trajectories.
    assert_true(
        abs(loss_fp32 - loss_bf16) < 0.15,
        "bf16 conv must track fp32 (got fp32="
        + String(loss_fp32)
        + ", bf16="
        + String(loss_bf16)
        + ")",
    )
    print("PASS")


def main() raises:
    test_conv2d_bf16_gpu()

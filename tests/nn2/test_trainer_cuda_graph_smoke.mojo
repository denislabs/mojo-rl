"""Smoke test: the nn2 Trainer's CUDA-graph-capturable device step.

Exercises `Trainer.load_fixed_batch` + `Trainer.train_step_device` both
eagerly and through `maybe_capture_replay` (the CUDA-graph capture harness),
overfitting one fixed mini-batch and asserting the loss drops and stays
finite.

On NVIDIA this validates real capture/replay (the first replay-loop call
captures the pure-device step, the rest replay it). On Apple / non-NVIDIA
`maybe_capture_replay` is a compile-time no-op, so the graph path runs
eagerly — bit-identical to the eager path, which is exactly what the final
`assert` checks (eager and graph reach the same loss).

Run (Apple):  pixi run -e apple  mojo run -I . tests/nn2/test_trainer_cuda_graph_smoke.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . tests/nn2/test_trainer_cuda_graph_smoke.mojo
"""

from std.math import exp, log, isfinite, abs
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.datasets import MNIST
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay


comptime IN_DIM = 784
comptime H = 128
comptime N_CLASSES = 10
comptime BATCH = 64
comptime N_STEPS = 150

comptime Net = Sequential[
    Linear[IN_DIM, H],
    ReLU[H],
    Linear[H, N_CLASSES],
]
comptime TRAINER = Trainer[
    Net, Adam, CrossEntropyLoss[N_CLASSES], BATCH, target="gpu"
]


def _ce_loss(
    logits: List[Scalar[DT]], targets_onehot: List[Scalar[DT]]
) -> Float64:
    """Mean cross-entropy over the batch from flat [BATCH, N_CLASSES] logits."""
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


def _make_batch(
    ds: MNIST, mut x: List[Scalar[DT]], mut y: List[Scalar[DT]]
):
    for i in range(BATCH * IN_DIM):
        x[i] = ds.train_images[i]
    for i in range(BATCH * N_CLASSES):
        y[i] = Scalar[DT](0.0)
    for i in range(BATCH):
        y[i * N_CLASSES + Int(ds.train_labels[i])] = Scalar[DT](1.0)


def _final_loss[
    USE_GRAPH: Bool
](
    ctx: DeviceContext,
    x: List[Scalar[DT]],
    y: List[Scalar[DT]],
) raises -> Float64:
    """Build a fresh trainer (seeded), overfit the fixed batch for N_STEPS
    via the device step (eager or graph), and return the final CE loss."""
    seed(123)
    var trainer = TRAINER.make[Kaiming](ctx)
    trainer.optim.lr = Scalar[DT](0.001)
    trainer.load_fixed_batch(x, y)

    comptime if USE_GRAPH:
        var graph: Optional[CUDAGraph] = None

        def _step() capturing raises -> None:
            trainer.train_step_device()

        for _ in range(N_STEPS):
            maybe_capture_replay[_step](graph, ctx)
    else:
        for _ in range(N_STEPS):
            trainer.train_step_device()
    ctx.synchronize()

    var out = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=Scalar[DT](0.0))
    trainer.predict(x, out)
    return _ce_loss(out, y)


def test_trainer_cuda_graph_smoke() raises:
    print("loading MNIST...")
    var ds = MNIST()
    var ctx = DeviceContext()

    var x = List[Scalar[DT]](length=BATCH * IN_DIM, fill=Scalar[DT](0.0))
    var y = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=Scalar[DT](0.0))
    _make_batch(ds, x, y)

    # Initial loss from an untrained, seeded trainer (predict only).
    seed(123)
    var t0 = TRAINER.make[Kaiming](ctx)
    t0.load_fixed_batch(x, y)
    var out0 = List[Scalar[DT]](length=BATCH * N_CLASSES, fill=Scalar[DT](0.0))
    t0.predict(x, out0)
    var loss_init = _ce_loss(out0, y)
    print("initial loss:", loss_init)

    var loss_eager = _final_loss[USE_GRAPH=False](ctx, x, y)
    print("eager final loss:", loss_eager)

    var loss_graph = _final_loss[USE_GRAPH=True](ctx, x, y)
    print("graph final loss:", loss_graph)

    assert_true(isfinite(loss_eager), "eager loss must be finite")
    assert_true(isfinite(loss_graph), "graph loss must be finite")
    assert_true(
        loss_eager < loss_init - 0.1,
        "eager device step must reduce the loss on a fixed batch",
    )
    assert_true(
        loss_graph < loss_init - 0.1,
        "graph device step must reduce the loss on a fixed batch",
    )
    # Same seed + same fixed batch + same step sequence ⇒ the graph path
    # (capture/replay on NVIDIA, no-op eager on Apple) must match the eager
    # path tightly.
    assert_true(
        abs(loss_eager - loss_graph) < 1e-3,
        "graph and eager paths must reach the same loss (got eager="
        + String(loss_eager)
        + ", graph="
        + String(loss_graph)
        + ")",
    )
    print("PASS")


def main() raises:
    test_trainer_cuda_graph_smoke()

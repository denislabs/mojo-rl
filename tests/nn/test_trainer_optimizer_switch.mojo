"""Trainer is generic over the Optimizer trait — switch Adam ↔ SGD.

A tiny synthetic 2-class problem trained by the SAME Trainer with two different
optimizers (`Trainer[..., Adam]` vs `Trainer[..., SGD]`), proving the Trainer's
`OPT: Optimizer` parameter lets you swap optimizers with no other change. Both
must train (loss finite + decreasing). CPU (fast, deterministic).

Run: pixi run mojo run -I . tests/nn/test_trainer_optimizer_switch.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.optimizer.sgd import SGD
from mojo_rl.nn.training.trainer import Trainer


comptime IN = 4
comptime H = 8
comptime NC = 2
comptime BATCH = 4
comptime N = 16  # samples
comptime EPOCHS = 30
comptime Net = Sequential[LinearReLU[IN, H], Linear[H, NC]]


def _data() -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    # Linearly-separable-ish: class = (x0 + x1 > 0). One-hot targets.
    var x = List[Scalar[DT]](length=N * IN, fill=Scalar[DT](0))
    var y = List[Scalar[DT]](length=N * NC, fill=Scalar[DT](0))
    for i in range(N):
        var a = Scalar[DT]((i % 5) - 2) * 0.5
        var b = Scalar[DT](((i * 3) % 7) - 3) * 0.4
        x[i * IN + 0] = a
        x[i * IN + 1] = b
        x[i * IN + 2] = Scalar[DT]((i % 3) - 1) * 0.3
        x[i * IN + 3] = Scalar[DT]((i % 2)) * 0.2
        var cls = 1 if (a + b) > Scalar[DT](0.0) else 0
        y[i * NC + cls] = Scalar[DT](1.0)
    return (x^, y^)


def _train[OPT: Optimizer](name: String, lr: Scalar[DT]) raises -> Bool:
    var d = _data()
    var tx = d[0].copy()  # independent owned Lists (avoid tuple-aliasing ref args)
    var ty = d[1].copy()
    var trainer = Trainer[Net, NC, IN, BATCH, "cpu", OPT=OPT].make[Kaiming](
        None, lr=lr
    )
    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for epoch in range(EPOCHS):
        var loss = trainer.train_epoch[N](tx, ty, None)
        if epoch == 0:
            first = loss
        last = loss
    print("  ", name, "loss", first, "->", last)
    # Switchability proof: it ran, loss is finite (x==x fails on NaN), and it
    # made progress.
    return (first == first) and (last == last) and last < first


def main() raises:
    print("Trainer optimizer switch (Adam vs SGD, same Trainer)")
    var adam_ok = _train[Adam]("Adam", Scalar[DT](1e-2))
    var sgd_ok = _train[SGD]("SGD ", Scalar[DT](1e-1))
    print("  Adam:", "OK" if adam_ok else "FAIL", " SGD:", "OK" if sgd_ok else "FAIL")
    assert_true(adam_ok and sgd_ok, "Trainer optimizer switch")
    print("OPTIMIZER SWITCH OK")

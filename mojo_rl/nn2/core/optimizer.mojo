"""Slim Optimizer trait.

Algorithm-specific hyperparams (lr, β₁, β₂, ε, weight_decay) are NOT
in the trait `make` signature; they live as public mut fields on each
concrete optimizer struct so external schedules can poke them without
rebuilding the optimizer.

Concrete usage:
    var opt = Adam.make[target="cpu", M=MyModel](model)
    opt.lr = Scalar[DT](3e-4)
    opt.beta1 = Scalar[DT](0.95)
    # ... train loop ...
    opt.step["cpu", M=MyModel](model)

Two factory overloads (CPU + GPU) mirror the existing convention.

The slim trait keeps `target: StaticString` and `M: Module` for full
parity with the existing surface — only the hyperparam args are dropped.
The `lr` mut-field pattern also means SAC alpha annealing / cosine
schedules can poke `opt.lr` in-place per-step without rebuilding the
optimizer.
"""

from std.gpu.host import DeviceContext

from ..constants import DT
from .module import Module


trait Optimizer(Defaultable & Movable & ImplicitlyDestructible):
    @staticmethod
    def make[target: StaticString, M: Module](mut model: M) raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M, ctx: DeviceContext,
    ) raises -> Self:
        ...

    def zero_grad[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        ...

    def step[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        ...

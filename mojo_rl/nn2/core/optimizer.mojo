"""Optimizer trait — `zero_grad` and `step` over a Module's parameter tree.

Phase 2.4: methods take `target: StaticString` as comptime method param.
`make[target, M]` factory declared so Trainer can dispatch via the bound.
"""

from std.gpu.host import DeviceContext

from ..constants import DT
from .module import Module


trait Optimizer(Defaultable & Movable & ImplicitlyDestructible):
    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
    ) raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString, M: Module](
        mut model: M,
        ctx: DeviceContext,
        lr: Scalar[DT] = 0.001,
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
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

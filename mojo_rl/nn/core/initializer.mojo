"""Initializer — storage-native weight/bias init (trait + structs).

The storage replacement for legacy `nn.core.initializer`: target-aware and
`Tensor`-based, with NO `UnsafePointer`. Each initializer fills the param's host
`.data` (host RNG → reproducible) and, on GPU, uploads to the device buffer —
exactly the legacy "host-fill then upload" recipe, but expressed over `Tensor`
and parametrised by `target` so one method serves CPU and GPU.

Leaves call these from the unified `make[target, INIT]` factory (init at
construction — there is no separate `reinit` pass). `init_bias` defaults to zero.

Structs:
- `Kaiming` : He-uniform  U[-sqrt(6/fan_in), +…]            (default for ReLU)
- `Xavier`  : Glorot-uniform U[-sqrt(6/(fan_in+fan_out)), …](default for Tanh)
- `Zero`    : all zeros
- `Normal[MEAN, STD]` : N(MEAN, STD) via Box-Muller         (GPT-2 init)
- `Deterministic` : the fixed `(i % 7 - 3) * 0.1` pattern — reproduces the old
                    leaf `_init_w` so parity spikes/gates stay bit-identical.
"""

from std.math import sqrt as fsqrt, log, cos, sin
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .tensor import Tensor


comptime _TWO_PI: Float64 = 6.283185307179586


trait Initializer:
    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        ...

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        ...


def _zero_bias[target: StaticString](
    mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
) raises:
    """Shared zero-bias fill (host) + upload on GPU."""
    for i in range(n):
        b.data[i] = Scalar[DT](0)
    comptime if target == "gpu":
        b.upload(ctx.value())


struct Kaiming(Initializer):
    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        var bound = fsqrt(6.0 / Float64(fan_in))
        for i in range(n):
            w.data[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * bound)
        comptime if target == "gpu":
            w.upload(ctx.value())

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        _zero_bias[target](b, n, ctx)


struct ScaledKaiming[NUM: Int, DEN: Int](Initializer):
    """He-uniform with the bound scaled by `NUM/DEN` (a rational so it is a
    comptime param). `ScaledKaiming[0, 1]` == `Zero`; `ScaledKaiming[1, 10]`
    keeps a tenth of the Kaiming magnitude — the "near-neutral but keep a little
    asymmetry" output-head init (e.g. positive-reward tasks), without the brittle
    scale-after-make surgery: wrap the leaf with `InitWith[Linear[...],
    ScaledKaiming[1, 10]]`."""

    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        var bound = fsqrt(6.0 / Float64(fan_in)) * (
            Float64(Self.NUM) / Float64(Self.DEN)
        )
        for i in range(n):
            w.data[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * bound)
        comptime if target == "gpu":
            w.upload(ctx.value())

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        _zero_bias[target](b, n, ctx)


struct Xavier(Initializer):
    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        var bound = fsqrt(6.0 / Float64(fan_in + fan_out))
        for i in range(n):
            w.data[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * bound)
        comptime if target == "gpu":
            w.upload(ctx.value())

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        _zero_bias[target](b, n, ctx)


struct Zero(Initializer):
    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(n):
            w.data[i] = Scalar[DT](0)
        comptime if target == "gpu":
            w.upload(ctx.value())

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        _zero_bias[target](b, n, ctx)


struct Normal[MEAN: Float64, STD: Float64](Initializer):
    """N(MEAN, STD) weights via Box-Muller (GPT-2 init = Normal[0.0, 0.02]).
    fan_in/fan_out accepted for trait conformance but ignored."""

    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        var i = 0
        while i < n:
            var u1 = random_float64()
            var u2 = random_float64()
            if u1 < 1e-12:
                u1 = 1e-12
            var r = fsqrt(-2.0 * log(u1))
            w.data[i] = Scalar[DT](Self.MEAN + Self.STD * r * cos(_TWO_PI * u2))
            i += 1
            if i < n:
                w.data[i] = Scalar[DT](
                    Self.MEAN + Self.STD * r * sin(_TWO_PI * u2)
                )
                i += 1
        comptime if target == "gpu":
            w.upload(ctx.value())

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        _zero_bias[target](b, n, ctx)


struct Deterministic(Initializer):
    """Fixed `(i % 7 - 3) * 0.1` weight pattern (bias 0) — reproduces the old
    leaf `_init_w`, so parity spikes/gates stay bit-identical. NOT for training
    (no randomness); training uses Kaiming/Xavier."""

    @staticmethod
    def init_weight[target: StaticString](
        mut w: Tensor, n: Int, fan_in: Int, fan_out: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(n):
            w.data[i] = Scalar[DT]((i % 7) - 3) * 0.1
        comptime if target == "gpu":
            w.upload(ctx.value())

    @staticmethod
    def init_bias[target: StaticString](
        mut b: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        _zero_bias[target](b, n, ctx)

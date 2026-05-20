"""Concrete weight initializers.

- `Kaiming`: He-uniform U[-sqrt(6/fan_in), sqrt(6/fan_in)]. Default for ReLU.
- `Xavier`:  Glorot-uniform U[-sqrt(6/(fan_in+fan_out)), …]. Default for Tanh/Sigmoid.
- `Zero`:    All zeros — for testing or as a placeholder bias.

All three set bias to 0 by default. The CrossEntropyLoss head conventionally
also wants zero bias.

These are HOST-SIDE fills. `Linear.make[INIT](ctx)` on GPU creates a
HostBuffer, runs the initializer over it, and uploads to device.
"""

from std.math import sqrt as fsqrt
from std.random import random_float64

from ..constants import DT
from ..core import Initializer


struct Kaiming(Initializer):
    @staticmethod
    def init_weight(
        buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
        fan_in: Int,
        fan_out: Int,
    ):
        var bound = fsqrt(6.0 / Float64(fan_in))
        for i in range(n_elems):
            var r = random_float64()       # [0, 1)
            buf[i] = Scalar[DT]((r * 2.0 - 1.0) * bound)

    @staticmethod
    def init_bias(
        buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ):
        for i in range(n_elems):
            buf[i] = 0.0


struct Xavier(Initializer):
    @staticmethod
    def init_weight(
        buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
        fan_in: Int,
        fan_out: Int,
    ):
        var bound = fsqrt(6.0 / Float64(fan_in + fan_out))
        for i in range(n_elems):
            var r = random_float64()
            buf[i] = Scalar[DT]((r * 2.0 - 1.0) * bound)

    @staticmethod
    def init_bias(
        buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ):
        for i in range(n_elems):
            buf[i] = 0.0


struct Zero(Initializer):
    """All zeros — primarily for unit tests."""

    @staticmethod
    def init_weight(
        buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
        fan_in: Int,
        fan_out: Int,
    ):
        for i in range(n_elems):
            buf[i] = 0.0

    @staticmethod
    def init_bias(
        buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ):
        for i in range(n_elems):
            buf[i] = 0.0

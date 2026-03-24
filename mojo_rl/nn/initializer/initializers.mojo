"""Weight initialization traits and implementations for neural networks.

Uses the MAX/Philox counter-based pattern: each element gets its own
RNG instance via PhiloxRandom(seed, offset=base+i), producing fully
independent streams. The base offset is derived from (FAN_IN, FAN_OUT)
so different layers in a model get different random sequences.

Usage:
    # In Trainer - initializer is a type parameter
    var trainer = Trainer[MODEL, OPTIMIZER, LOSS, Xavier](
        model, optimizer, loss, Xavier()
    )

    # Or with Kaiming for ReLU networks
    var trainer = Trainer[MODEL, OPTIMIZER, LOSS, Kaiming](
        model, optimizer, loss, Kaiming()
    )
"""
from layout import LayoutTensor, Layout
from ..constants import dtype
from std.math import sqrt, log, cos, sin, pi
from std.random.philox import Random as PhiloxRandom


def _layer_offset[FAN_IN: Int, FAN_OUT: Int]() -> UInt64:
    """Derive a unique base offset from layer dimensions.

    Uses large primes so layers with different (FAN_IN, FAN_OUT)
    get non-overlapping RNG streams without any signature changes.
    """
    return UInt64(FAN_IN) * 1000003 + UInt64(FAN_OUT) * 999983


trait Initializer(Copyable & Movable & ImplicitlyCopyable):
    """Base trait for weight initializers.

    Initializers are used to set initial values for model parameters.
    Different initialization strategies are optimal for different
    activation functions and network architectures.
    """

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        """Initialize parameters.

        Parameters:
            SIZE: Total number of parameters to initialize.
            FAN_IN: Number of input features (used by some initializers).
            FAN_OUT: Number of output features (used by some initializers).

        Args:
            params: LayoutTensor to initialize.
        """
        ...


struct Xavier[SEED: UInt64 = 0](Initializer):
    """Xavier/Glorot initialization.

    Weights are drawn from U(-limit, limit) where limit = sqrt(6/(fan_in+fan_out)).

    This is optimal for linear activations and works well for tanh/sigmoid.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var limit = sqrt(6.0 / Scalar[dtype](FAN_IN + FAN_OUT))
        var base = _layer_offset[FAN_IN, FAN_OUT]()
        for i in range(SIZE):
            var rng = PhiloxRandom(seed=Self.SEED, offset=base + UInt64(i))
            var val = rng.step_uniform()
            params[i] = Scalar[dtype]((val[0] * 2.0 - 1.0) * limit)


struct Kaiming[SEED: UInt64 = 0](Initializer):
    """Kaiming/He initialization.

    Weights are drawn from U(-limit, limit) where limit = sqrt(6/fan_in).

    This is optimal for ReLU activations, accounting for the fact that
    ReLU zeros out half the distribution.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var limit = sqrt(6.0 / Scalar[dtype](FAN_IN))
        var base = _layer_offset[FAN_IN, FAN_OUT]()
        for i in range(SIZE):
            var rng = PhiloxRandom(seed=Self.SEED, offset=base + UInt64(i))
            var val = rng.step_uniform()
            params[i] = Scalar[dtype]((val[0] * 2.0 - 1.0) * limit)


struct LeCun[SEED: UInt64 = 0](Initializer):
    """LeCun initialization.

    Weights are drawn from U(-limit, limit) where limit = sqrt(3/fan_in).

    This is the original initialization proposed by LeCun for
    networks with tanh activations.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var limit = sqrt(3.0 / Scalar[dtype](FAN_IN))
        var base = _layer_offset[FAN_IN, FAN_OUT]()
        for i in range(SIZE):
            var rng = PhiloxRandom(seed=Self.SEED, offset=base + UInt64(i))
            var val = rng.step_uniform()
            params[i] = Scalar[dtype]((val[0] * 2.0 - 1.0) * limit)


struct Zeros(Initializer):
    """Initialize all parameters to zero."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        for i in range(SIZE):
            params[i] = 0


struct Ones(Initializer):
    """Initialize all parameters to one."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        for i in range(SIZE):
            params[i] = 1


struct Constant[VALUE: Scalar[dtype]](Initializer):
    """Initialize all parameters to a constant value."""

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        for i in range(SIZE):
            params[i] = Self.VALUE


struct Uniform[LOW: Float64, HIGH: Float64, SEED: UInt64 = 0](Initializer):
    """Initialize parameters from uniform distribution U(low, high)."""

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var range_val = Scalar[dtype](Self.HIGH - Self.LOW)
        var base = _layer_offset[FAN_IN, FAN_OUT]()
        for i in range(SIZE):
            var rng = PhiloxRandom(seed=Self.SEED, offset=base + UInt64(i))
            var val = rng.step_uniform()
            params[i] = Scalar[dtype](
                val[0] * range_val + Scalar[dtype](Self.LOW)
            )


struct Normal[MEAN: Float64, STD: Float64, SEED: UInt64 = 0](Initializer):
    """Initialize parameters from normal distribution N(mean, std).

    Uses Box-Muller transform on Philox uniform pairs.
    """

    @staticmethod
    def init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var base = _layer_offset[FAN_IN, FAN_OUT]()
        # Box-Muller: each pair of uniforms → 2 normals
        var i = 0
        var pair_idx: UInt64 = 0
        while i < SIZE:
            # Two independent uniform streams for each Box-Muller pair
            var rng1 = PhiloxRandom(
                seed=Self.SEED, offset=base + pair_idx * 2
            )
            var rng2 = PhiloxRandom(
                seed=Self.SEED, offset=base + pair_idx * 2 + 1
            )
            var u1 = rng1.step_uniform()[0]
            var u2 = rng2.step_uniform()[0]
            pair_idx += 1

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            var z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
            params[i] = Scalar[dtype](
                z0 * Scalar[dtype](Self.STD) + Scalar[dtype](Self.MEAN)
            )
            i += 1

            # Use the second value if we have space
            if i < SIZE:
                var z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * pi * u2)
                params[i] = Scalar[dtype](
                    z1 * Scalar[dtype](Self.STD) + Scalar[dtype](Self.MEAN)
                )
                i += 1

"""Weight initialization traits and implementations for neural networks.

This module provides a trait-based initialization system:
- Initializer trait: Base interface for all initializers
- Xavier/Glorot: Good for tanh/sigmoid activations
- Kaiming/He: Good for ReLU activations
- Zeros, Ones, Constant: Simple initializers
- Uniform, Normal: Distribution-based initializers

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
from math import sqrt, log, cos, sin, pi
from random.philox import Random as PhiloxRandom


trait Initializer(Copyable & Movable & ImplicitlyCopyable):
    """Base trait for weight initializers.

    Initializers are used to set initial values for model parameters.
    Different initialization strategies are optimal for different
    activation functions and network architectures.
    """

    @staticmethod
    fn init[
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

    Weights are drawn from U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    or equivalently scaled normal distribution.

    This is optimal for linear activations and works well for tanh/sigmoid.
    """

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var rng = PhiloxRandom(seed=Self.SEED, offset=0)
        var rand_vals = rng.step_uniform()
        var std = sqrt(2.0 / Scalar[dtype](FAN_IN + FAN_OUT))
        for i in range(SIZE):
            params[i] = Scalar[dtype]((rand_vals[i] * 2.0 - 1.0) * std)


struct Kaiming[SEED: UInt64 = 0](Initializer):
    """Kaiming/He initialization.

    Weights are drawn from N(0, sqrt(2/fan_in)).

    This is optimal for ReLU activations, accounting for the fact that
    ReLU zeros out half the distribution.
    """

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var std = sqrt(2.0 / Scalar[dtype](FAN_IN))
        var rng = PhiloxRandom(seed=Self.SEED, offset=0)
        var rand_vals = rng.step_uniform()
        for i in range(SIZE):
            params[i] = Scalar[dtype]((rand_vals[i] * 2.0 - 1.0) * std)


struct LeCun[SEED: UInt64 = 0](Initializer):
    """LeCun initialization.

    Weights are drawn from N(0, sqrt(1/fan_in)).

    This is the original initialization proposed by LeCun for
    networks with tanh activations.
    """

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var std = sqrt(1.0 / Scalar[dtype](FAN_IN))
        var rng = PhiloxRandom(seed=Self.SEED, offset=0)
        var rand_vals = rng.step_uniform()
        for i in range(SIZE):
            params[i] = Scalar[dtype]((rand_vals[i] * 2.0 - 1.0) * std)


struct Zeros(Initializer):
    """Initialize all parameters to zero.

    Useful for biases or when you want to start from a clean slate.
    Note: Using zeros for weights will cause issues with gradient flow.
    """

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        for i in range(SIZE):
            params[i] = 0


struct Ones(Initializer):
    """Initialize all parameters to one."""

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        for i in range(SIZE):
            params[i] = 1


struct Constant[VALUE: Scalar[dtype]](Initializer):
    """Initialize all parameters to a constant value."""

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        for i in range(SIZE):
            params[i] = Self.VALUE


struct Uniform[LOW: Float64, HIGH: Float64, SEED: UInt64 = 0](Initializer):
    """Initialize parameters from uniform distribution U(low, high)."""

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var range_val = Scalar[dtype](Self.HIGH - Self.LOW)
        var rng = PhiloxRandom(seed=Self.SEED, offset=0)
        var rand_vals = rng.step_uniform()
        for i in range(SIZE):
            params[i] = Scalar[dtype](rand_vals[i] * range_val + Self.LOW)


struct Normal[MEAN: Float64, STD: Float64, SEED: UInt64 = 0](Initializer):
    """Initialize parameters from normal distribution N(mean, std).

    Uses Box-Muller transform to generate normal random numbers.
    """

    @staticmethod
    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](mut params: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin]):
        var rng = PhiloxRandom(seed=Self.SEED, offset=0)
        var rand_vals = rng.step_uniform()
        # Box-Muller transform generates pairs of normal random numbers
        var i = 0
        while i < SIZE:
            var u1 = rand_vals[i]
            var u2 = rand_vals[i + 1]

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            var z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
            params[i] = Scalar[dtype](z0 * Self.STD + Self.MEAN)
            i += 1

            # Use the second value if we have space
            if i < SIZE:
                var z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * pi * u2)
                params[i] = Scalar[dtype](z1 * Self.STD + Self.MEAN)
                i += 1

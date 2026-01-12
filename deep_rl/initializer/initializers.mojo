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

from ..constants import dtype
from math import sqrt
from random import random_float64


trait Initializer(Copyable & Movable & ImplicitlyCopyable):
    """Base trait for weight initializers.

    Initializers are used to set initial values for model parameters.
    Different initialization strategies are optimal for different
    activation functions and network architectures.
    """

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        """Initialize parameters.

        Parameters:
            SIZE: Total number of parameters to initialize.
            FAN_IN: Number of input features (used by some initializers).
            FAN_OUT: Number of output features (used by some initializers).

        Returns:
            Initialized parameter array.
        """
        ...


struct Xavier(Initializer):
    """Xavier/Glorot initialization.

    Weights are drawn from U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    or equivalently scaled normal distribution.

    This is optimal for linear activations and works well for tanh/sigmoid.
    """

    fn __init__(out self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        var std = sqrt(2.0 / Float64(FAN_IN + FAN_OUT))
        for i in range(SIZE):
            params[i] = Scalar[dtype]((random_float64() * 2.0 - 1.0) * std)
        return params^


struct Kaiming(Initializer):
    """Kaiming/He initialization.

    Weights are drawn from N(0, sqrt(2/fan_in)).

    This is optimal for ReLU activations, accounting for the fact that
    ReLU zeros out half the distribution.
    """

    fn __init__(out self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        var std = sqrt(2.0 / Float64(FAN_IN))
        for i in range(SIZE):
            params[i] = Scalar[dtype]((random_float64() * 2.0 - 1.0) * std)
        return params^


struct LeCun(Initializer):
    """LeCun initialization.

    Weights are drawn from N(0, sqrt(1/fan_in)).

    This is the original initialization proposed by LeCun for
    networks with tanh activations.
    """

    fn __init__(out self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        var std = sqrt(1.0 / Float64(FAN_IN))
        for i in range(SIZE):
            params[i] = Scalar[dtype]((random_float64() * 2.0 - 1.0) * std)
        return params^


struct Zeros(Initializer):
    """Initialize all parameters to zero.

    Useful for biases or when you want to start from a clean slate.
    Note: Using zeros for weights will cause issues with gradient flow.
    """

    fn __init__(out self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        for i in range(SIZE):
            params[i] = 0
        return params^


struct Ones(Initializer):
    """Initialize all parameters to one."""

    fn __init__(out self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        for i in range(SIZE):
            params[i] = 1
        return params^


struct Constant(Initializer):
    """Initialize all parameters to a constant value."""

    var value: Scalar[dtype]

    fn __init__(out self, value: Scalar[dtype] = 0):
        self.value = value

    fn __copyinit__(out self, other: Self):
        self.value = other.value

    fn __moveinit__(out self, deinit other: Self):
        self.value = other.value

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        for i in range(SIZE):
            params[i] = self.value
        return params^


struct Uniform(Initializer):
    """Initialize parameters from uniform distribution U(low, high)."""

    var low: Float64
    var high: Float64

    fn __init__(out self, low: Float64 = -1.0, high: Float64 = 1.0):
        self.low = low
        self.high = high

    fn __copyinit__(out self, other: Self):
        self.low = other.low
        self.high = other.high

    fn __moveinit__(out self, deinit other: Self):
        self.low = other.low
        self.high = other.high

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        var range_val = self.high - self.low
        for i in range(SIZE):
            params[i] = Scalar[dtype](random_float64() * range_val + self.low)
        return params^


struct Normal(Initializer):
    """Initialize parameters from normal distribution N(mean, std).

    Uses Box-Muller transform to generate normal random numbers.
    """

    var mean: Float64
    var std: Float64

    fn __init__(out self, mean: Float64 = 0.0, std: Float64 = 1.0):
        self.mean = mean
        self.std = std

    fn __copyinit__(out self, other: Self):
        self.mean = other.mean
        self.std = other.std

    fn __moveinit__(out self, deinit other: Self):
        self.mean = other.mean
        self.std = other.std

    fn init[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int
    ](self) -> InlineArray[Scalar[dtype], SIZE]:
        from math import log, cos, sin

        var params = InlineArray[Scalar[dtype], SIZE](uninitialized=True)
        var pi = 3.14159265358979323846

        # Box-Muller transform generates pairs of normal random numbers
        var i = 0
        while i < SIZE:
            var u1 = random_float64()
            var u2 = random_float64()

            # Avoid log(0)
            if u1 < 1e-10:
                u1 = 1e-10

            var z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
            params[i] = Scalar[dtype](z0 * self.std + self.mean)
            i += 1

            # Use the second value if we have space
            if i < SIZE:
                var z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * pi * u2)
                params[i] = Scalar[dtype](z1 * self.std + self.mean)
                i += 1

        return params^

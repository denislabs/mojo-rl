"""Parameter space definitions for hyperparameter search.

This module provides structs for defining hyperparameter search spaces,
including support for both grid search (discrete enumeration) and random
search (continuous sampling).
"""

from random import random_float64, seed
from math import log, exp


# ============================================================================
# Parameter Value Types
# ============================================================================


struct FloatParam(Copyable, Movable, ImplicitlyCopyable):
    """A single float parameter with its search space.

    Supports both linear and logarithmic scaling for sampling.
    Log scaling is useful for parameters like learning rate where
    the relative difference matters more than absolute difference.
    """

    var name: String
    var min_val: Float64
    var max_val: Float64
    var log_scale: Bool
    var num_values: Int

    fn __init__(
        out self,
        name: String,
        min_val: Float64,
        max_val: Float64,
        log_scale: Bool = False,
        num_values: Int = 5,
    ):
        """Initialize a float parameter.

        Args:
            name: Parameter name for display/export.
            min_val: Minimum value (must be > 0 for log scale).
            max_val: Maximum value.
            log_scale: If True, sample/enumerate in log space.
            num_values: Number of discrete values for grid search.
        """
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.log_scale = log_scale
        self.num_values = num_values

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.min_val = existing.min_val
        self.max_val = existing.max_val
        self.log_scale = existing.log_scale
        self.num_values = existing.num_values

    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.min_val = existing.min_val
        self.max_val = existing.max_val
        self.log_scale = existing.log_scale
        self.num_values = existing.num_values

    fn sample_random(self) -> Float64:
        """Sample a random value from the parameter space."""
        var rand = random_float64()
        if self.log_scale:
            var log_min = log(self.min_val)
            var log_max = log(self.max_val)
            return exp(log_min + rand * (log_max - log_min))
        else:
            return self.min_val + rand * (self.max_val - self.min_val)

    fn get_grid_values(self) -> List[Float64]:
        """Get discrete grid values for grid search."""
        var values = List[Float64]()
        if self.num_values <= 1:
            values.append(self.min_val)
            return values^

        if self.log_scale:
            var log_min = log(self.min_val)
            var log_max = log(self.max_val)
            for i in range(self.num_values):
                var t = Float64(i) / Float64(self.num_values - 1)
                values.append(exp(log_min + t * (log_max - log_min)))
        else:
            for i in range(self.num_values):
                var t = Float64(i) / Float64(self.num_values - 1)
                values.append(self.min_val + t * (self.max_val - self.min_val))
        return values^

    fn get_value_at_index(self, idx: Int) -> Float64:
        """Get the grid value at a specific index."""
        if self.num_values <= 1:
            return self.min_val
        var t = Float64(idx) / Float64(self.num_values - 1)
        if self.log_scale:
            var log_min = log(self.min_val)
            var log_max = log(self.max_val)
            return exp(log_min + t * (log_max - log_min))
        else:
            return self.min_val + t * (self.max_val - self.min_val)


struct IntParam(Copyable, Movable, ImplicitlyCopyable):
    """A single integer parameter with its search space."""

    var name: String
    var values: List[Int]

    fn __init__(out self, name: String, min_val: Int, max_val: Int):
        """Initialize with a range of consecutive integers."""
        self.name = name
        self.values = List[Int]()
        for i in range(min_val, max_val + 1):
            self.values.append(i)

    fn __init__(out self, name: String, var values: List[Int]):
        """Initialize with an explicit list of values."""
        self.name = name
        self.values = values^

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.values = List[Int]()
        for i in range(len(existing.values)):
            self.values.append(existing.values[i])

    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.values = existing.values^

    fn sample_random(self) -> Int:
        """Sample a random value from the parameter space."""
        var idx = Int(random_float64() * Float64(len(self.values)))
        if idx >= len(self.values):
            idx = len(self.values) - 1
        return self.values[idx]

    fn get_grid_values(self) -> List[Int]:
        """Get all values for grid search."""
        var result = List[Int]()
        for i in range(len(self.values)):
            result.append(self.values[i])
        return result^

    fn num_values(self) -> Int:
        """Return the number of discrete values."""
        return len(self.values)

    fn get_value_at_index(self, idx: Int) -> Int:
        """Get the value at a specific index."""
        return self.values[idx]


struct BoolParam(Copyable, Movable):
    """A boolean parameter."""

    var name: String

    fn __init__(out self, name: String):
        self.name = name

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name

    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^

    fn sample_random(self) -> Bool:
        """Sample a random boolean value."""
        return random_float64() < 0.5

    fn get_grid_values(self) -> List[Bool]:
        """Get both boolean values for grid search."""
        var values = List[Bool]()
        values.append(False)
        values.append(True)
        return values^


# ============================================================================
# Hyperparameter Configuration Structs
# ============================================================================


struct TabularHyperparams(Copyable, Movable):
    """Hyperparameters for tabular agents (Q-Learning, SARSA, etc.)."""

    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min

    fn to_csv_header(self) -> String:
        """Return CSV header for hyperparameters."""
        return "learning_rate,discount_factor,epsilon,epsilon_decay,epsilon_min"

    fn to_csv_row(self) -> String:
        """Return CSV row for hyperparameters."""
        return (
            String(self.learning_rate)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.epsilon)
            + ","
            + String(self.epsilon_decay)
            + ","
            + String(self.epsilon_min)
        )


struct TabularParamSpace(Copyable, Movable):
    """Search space for tabular agents."""

    var learning_rate: FloatParam
    var discount_factor: FloatParam
    var epsilon: FloatParam
    var epsilon_decay: FloatParam
    var epsilon_min: FloatParam

    fn __init__(out self):
        """Initialize with default search ranges."""
        self.learning_rate = FloatParam(
            "learning_rate", 0.01, 0.5, log_scale=True, num_values=5
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.9, 0.999, log_scale=False, num_values=4
        )
        self.epsilon = FloatParam(
            "epsilon", 0.5, 1.0, log_scale=False, num_values=3
        )
        self.epsilon_decay = FloatParam(
            "epsilon_decay", 0.99, 0.999, log_scale=False, num_values=3
        )
        self.epsilon_min = FloatParam(
            "epsilon_min", 0.01, 0.1, log_scale=False, num_values=2
        )

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate^
        self.discount_factor = existing.discount_factor^
        self.epsilon = existing.epsilon^
        self.epsilon_decay = existing.epsilon_decay^
        self.epsilon_min = existing.epsilon_min^

    fn sample_random(self) -> TabularHyperparams:
        """Sample random hyperparameters from the search space."""
        return TabularHyperparams(
            learning_rate=self.learning_rate.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            epsilon=self.epsilon.sample_random(),
            epsilon_decay=self.epsilon_decay.sample_random(),
            epsilon_min=self.epsilon_min.sample_random(),
        )

    fn get_grid_size(self) -> Int:
        """Return total number of grid combinations."""
        return (
            self.learning_rate.num_values
            * self.discount_factor.num_values
            * self.epsilon.num_values
            * self.epsilon_decay.num_values
            * self.epsilon_min.num_values
        )

    fn get_grid_config(self, index: Int) -> TabularHyperparams:
        """Get hyperparameters for a specific grid index.

        Decodes the flat index into multi-dimensional indices.
        """
        var remaining = index

        # Decode indices (last dimension changes fastest)
        var lr_idx = remaining % self.learning_rate.num_values
        remaining //= self.learning_rate.num_values

        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values

        var eps_idx = remaining % self.epsilon.num_values
        remaining //= self.epsilon.num_values

        var decay_idx = remaining % self.epsilon_decay.num_values
        remaining //= self.epsilon_decay.num_values

        var min_idx = remaining % self.epsilon_min.num_values

        return TabularHyperparams(
            learning_rate=self.learning_rate.get_value_at_index(lr_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            epsilon=self.epsilon.get_value_at_index(eps_idx),
            epsilon_decay=self.epsilon_decay.get_value_at_index(decay_idx),
            epsilon_min=self.epsilon_min.get_value_at_index(min_idx),
        )


# ============================================================================
# N-Step Hyperparameters
# ============================================================================


struct NStepHyperparams(Copyable, Movable):
    """Hyperparameters for n-step agents (N-step SARSA)."""

    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var n: Int

    fn __init__(
        out self,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        n: Int = 3,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n = n

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.n = existing.n

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.n = existing.n

    fn to_csv_header(self) -> String:
        return "learning_rate,discount_factor,epsilon,epsilon_decay,epsilon_min,n"

    fn to_csv_row(self) -> String:
        return (
            String(self.learning_rate)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.epsilon)
            + ","
            + String(self.epsilon_decay)
            + ","
            + String(self.epsilon_min)
            + ","
            + String(self.n)
        )


struct NStepParamSpace(Copyable, Movable):
    """Search space for n-step agents."""

    var learning_rate: FloatParam
    var discount_factor: FloatParam
    var epsilon: FloatParam
    var epsilon_decay: FloatParam
    var epsilon_min: FloatParam
    var n: IntParam

    fn __init__(out self):
        self.learning_rate = FloatParam(
            "learning_rate", 0.01, 0.5, log_scale=True, num_values=5
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.9, 0.999, log_scale=False, num_values=4
        )
        self.epsilon = FloatParam("epsilon", 0.5, 1.0, log_scale=False, num_values=3)
        self.epsilon_decay = FloatParam(
            "epsilon_decay", 0.99, 0.999, log_scale=False, num_values=3
        )
        self.epsilon_min = FloatParam(
            "epsilon_min", 0.01, 0.1, log_scale=False, num_values=2
        )
        # Common n values: 1, 3, 5, 10
        var n_values = List[Int]()
        n_values.append(1)
        n_values.append(3)
        n_values.append(5)
        n_values.append(10)
        self.n = IntParam("n", n_values^)

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.n = existing.n

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate^
        self.discount_factor = existing.discount_factor^
        self.epsilon = existing.epsilon^
        self.epsilon_decay = existing.epsilon_decay^
        self.epsilon_min = existing.epsilon_min^
        self.n = existing.n^

    fn sample_random(self) -> NStepHyperparams:
        return NStepHyperparams(
            learning_rate=self.learning_rate.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            epsilon=self.epsilon.sample_random(),
            epsilon_decay=self.epsilon_decay.sample_random(),
            epsilon_min=self.epsilon_min.sample_random(),
            n=self.n.sample_random(),
        )

    fn get_grid_size(self) -> Int:
        return (
            self.learning_rate.num_values
            * self.discount_factor.num_values
            * self.epsilon.num_values
            * self.epsilon_decay.num_values
            * self.epsilon_min.num_values
            * self.n.num_values()
        )

    fn get_grid_config(self, index: Int) -> NStepHyperparams:
        var remaining = index
        var lr_idx = remaining % self.learning_rate.num_values
        remaining //= self.learning_rate.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var eps_idx = remaining % self.epsilon.num_values
        remaining //= self.epsilon.num_values
        var decay_idx = remaining % self.epsilon_decay.num_values
        remaining //= self.epsilon_decay.num_values
        var min_idx = remaining % self.epsilon_min.num_values
        remaining //= self.epsilon_min.num_values
        var n_idx = remaining % self.n.num_values()

        return NStepHyperparams(
            learning_rate=self.learning_rate.get_value_at_index(lr_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            epsilon=self.epsilon.get_value_at_index(eps_idx),
            epsilon_decay=self.epsilon_decay.get_value_at_index(decay_idx),
            epsilon_min=self.epsilon_min.get_value_at_index(min_idx),
            n=self.n.get_value_at_index(n_idx),
        )


# ============================================================================
# Lambda (Eligibility Trace) Hyperparameters
# ============================================================================


struct LambdaHyperparams(Copyable, Movable):
    """Hyperparameters for eligibility trace agents (SARSA-lambda)."""

    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var lambda_: Float64

    fn __init__(
        out self,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        lambda_: Float64 = 0.9,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lambda_ = lambda_

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.lambda_ = existing.lambda_

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.lambda_ = existing.lambda_

    fn to_csv_header(self) -> String:
        return "learning_rate,discount_factor,epsilon,epsilon_decay,epsilon_min,lambda"

    fn to_csv_row(self) -> String:
        return (
            String(self.learning_rate)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.epsilon)
            + ","
            + String(self.epsilon_decay)
            + ","
            + String(self.epsilon_min)
            + ","
            + String(self.lambda_)
        )


struct LambdaParamSpace(Copyable, Movable):
    """Search space for eligibility trace agents."""

    var learning_rate: FloatParam
    var discount_factor: FloatParam
    var epsilon: FloatParam
    var epsilon_decay: FloatParam
    var epsilon_min: FloatParam
    var lambda_: FloatParam

    fn __init__(out self):
        self.learning_rate = FloatParam(
            "learning_rate", 0.01, 0.5, log_scale=True, num_values=5
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.9, 0.999, log_scale=False, num_values=4
        )
        self.epsilon = FloatParam("epsilon", 0.5, 1.0, log_scale=False, num_values=3)
        self.epsilon_decay = FloatParam(
            "epsilon_decay", 0.99, 0.999, log_scale=False, num_values=3
        )
        self.epsilon_min = FloatParam(
            "epsilon_min", 0.01, 0.1, log_scale=False, num_values=2
        )
        self.lambda_ = FloatParam("lambda", 0.8, 0.99, log_scale=False, num_values=4)

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.lambda_ = existing.lambda_

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate^
        self.discount_factor = existing.discount_factor^
        self.epsilon = existing.epsilon^
        self.epsilon_decay = existing.epsilon_decay^
        self.epsilon_min = existing.epsilon_min^
        self.lambda_ = existing.lambda_^

    fn sample_random(self) -> LambdaHyperparams:
        return LambdaHyperparams(
            learning_rate=self.learning_rate.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            epsilon=self.epsilon.sample_random(),
            epsilon_decay=self.epsilon_decay.sample_random(),
            epsilon_min=self.epsilon_min.sample_random(),
            lambda_=self.lambda_.sample_random(),
        )

    fn get_grid_size(self) -> Int:
        return (
            self.learning_rate.num_values
            * self.discount_factor.num_values
            * self.epsilon.num_values
            * self.epsilon_decay.num_values
            * self.epsilon_min.num_values
            * self.lambda_.num_values
        )

    fn get_grid_config(self, index: Int) -> LambdaHyperparams:
        var remaining = index
        var lr_idx = remaining % self.learning_rate.num_values
        remaining //= self.learning_rate.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var eps_idx = remaining % self.epsilon.num_values
        remaining //= self.epsilon.num_values
        var decay_idx = remaining % self.epsilon_decay.num_values
        remaining //= self.epsilon_decay.num_values
        var min_idx = remaining % self.epsilon_min.num_values
        remaining //= self.epsilon_min.num_values
        var lambda_idx = remaining % self.lambda_.num_values

        return LambdaHyperparams(
            learning_rate=self.learning_rate.get_value_at_index(lr_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            epsilon=self.epsilon.get_value_at_index(eps_idx),
            epsilon_decay=self.epsilon_decay.get_value_at_index(decay_idx),
            epsilon_min=self.epsilon_min.get_value_at_index(min_idx),
            lambda_=self.lambda_.get_value_at_index(lambda_idx),
        )


# ============================================================================
# Model-Based Hyperparameters
# ============================================================================


struct ModelBasedHyperparams(Copyable, Movable):
    """Hyperparameters for model-based agents (Dyna-Q, Priority Sweeping)."""

    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var n_planning: Int

    fn __init__(
        out self,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        n_planning: Int = 5,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_planning = n_planning

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.n_planning = existing.n_planning

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.n_planning = existing.n_planning

    fn to_csv_header(self) -> String:
        return "learning_rate,discount_factor,epsilon,epsilon_decay,epsilon_min,n_planning"

    fn to_csv_row(self) -> String:
        return (
            String(self.learning_rate)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.epsilon)
            + ","
            + String(self.epsilon_decay)
            + ","
            + String(self.epsilon_min)
            + ","
            + String(self.n_planning)
        )


struct ModelBasedParamSpace(Copyable, Movable):
    """Search space for model-based agents."""

    var learning_rate: FloatParam
    var discount_factor: FloatParam
    var epsilon: FloatParam
    var epsilon_decay: FloatParam
    var epsilon_min: FloatParam
    var n_planning: IntParam

    fn __init__(out self):
        self.learning_rate = FloatParam(
            "learning_rate", 0.01, 0.5, log_scale=True, num_values=5
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.9, 0.999, log_scale=False, num_values=4
        )
        self.epsilon = FloatParam("epsilon", 0.5, 1.0, log_scale=False, num_values=3)
        self.epsilon_decay = FloatParam(
            "epsilon_decay", 0.99, 0.999, log_scale=False, num_values=3
        )
        self.epsilon_min = FloatParam(
            "epsilon_min", 0.01, 0.1, log_scale=False, num_values=2
        )
        # Common n_planning values
        var n_values = List[Int]()
        n_values.append(5)
        n_values.append(10)
        n_values.append(20)
        n_values.append(50)
        self.n_planning = IntParam("n_planning", n_values^)

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.n_planning = existing.n_planning

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate^
        self.discount_factor = existing.discount_factor^
        self.epsilon = existing.epsilon^
        self.epsilon_decay = existing.epsilon_decay^
        self.epsilon_min = existing.epsilon_min^
        self.n_planning = existing.n_planning^

    fn sample_random(self) -> ModelBasedHyperparams:
        return ModelBasedHyperparams(
            learning_rate=self.learning_rate.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            epsilon=self.epsilon.sample_random(),
            epsilon_decay=self.epsilon_decay.sample_random(),
            epsilon_min=self.epsilon_min.sample_random(),
            n_planning=self.n_planning.sample_random(),
        )

    fn get_grid_size(self) -> Int:
        return (
            self.learning_rate.num_values
            * self.discount_factor.num_values
            * self.epsilon.num_values
            * self.epsilon_decay.num_values
            * self.epsilon_min.num_values
            * self.n_planning.num_values()
        )

    fn get_grid_config(self, index: Int) -> ModelBasedHyperparams:
        var remaining = index
        var lr_idx = remaining % self.learning_rate.num_values
        remaining //= self.learning_rate.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var eps_idx = remaining % self.epsilon.num_values
        remaining //= self.epsilon.num_values
        var decay_idx = remaining % self.epsilon_decay.num_values
        remaining //= self.epsilon_decay.num_values
        var min_idx = remaining % self.epsilon_min.num_values
        remaining //= self.epsilon_min.num_values
        var n_idx = remaining % self.n_planning.num_values()

        return ModelBasedHyperparams(
            learning_rate=self.learning_rate.get_value_at_index(lr_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            epsilon=self.epsilon.get_value_at_index(eps_idx),
            epsilon_decay=self.epsilon_decay.get_value_at_index(decay_idx),
            epsilon_min=self.epsilon_min.get_value_at_index(min_idx),
            n_planning=self.n_planning.get_value_at_index(n_idx),
        )


# ============================================================================
# Replay Buffer Hyperparameters
# ============================================================================


struct ReplayHyperparams(Copyable, Movable):
    """Hyperparameters for replay-based agents."""

    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var buffer_size: Int
    var batch_size: Int
    var min_buffer_size: Int

    fn __init__(
        out self,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        buffer_size: Int = 1000,
        batch_size: Int = 32,
        min_buffer_size: Int = 100,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.buffer_size = existing.buffer_size
        self.batch_size = existing.batch_size
        self.min_buffer_size = existing.min_buffer_size

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.buffer_size = existing.buffer_size
        self.batch_size = existing.batch_size
        self.min_buffer_size = existing.min_buffer_size

    fn to_csv_header(self) -> String:
        return "learning_rate,discount_factor,epsilon,epsilon_decay,epsilon_min,buffer_size,batch_size,min_buffer_size"

    fn to_csv_row(self) -> String:
        return (
            String(self.learning_rate)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.epsilon)
            + ","
            + String(self.epsilon_decay)
            + ","
            + String(self.epsilon_min)
            + ","
            + String(self.buffer_size)
            + ","
            + String(self.batch_size)
            + ","
            + String(self.min_buffer_size)
        )


struct ReplayParamSpace(Copyable, Movable):
    """Search space for replay-based agents."""

    var learning_rate: FloatParam
    var discount_factor: FloatParam
    var epsilon: FloatParam
    var epsilon_decay: FloatParam
    var epsilon_min: FloatParam
    var buffer_size: IntParam
    var batch_size: IntParam

    fn __init__(out self):
        self.learning_rate = FloatParam(
            "learning_rate", 0.01, 0.5, log_scale=True, num_values=5
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.9, 0.999, log_scale=False, num_values=4
        )
        self.epsilon = FloatParam("epsilon", 0.5, 1.0, log_scale=False, num_values=3)
        self.epsilon_decay = FloatParam(
            "epsilon_decay", 0.99, 0.999, log_scale=False, num_values=3
        )
        self.epsilon_min = FloatParam(
            "epsilon_min", 0.01, 0.1, log_scale=False, num_values=2
        )
        # Buffer sizes
        var buffer_values = List[Int]()
        buffer_values.append(500)
        buffer_values.append(1000)
        buffer_values.append(5000)
        buffer_values.append(10000)
        self.buffer_size = IntParam("buffer_size", buffer_values^)
        # Batch sizes
        var batch_values = List[Int]()
        batch_values.append(16)
        batch_values.append(32)
        batch_values.append(64)
        self.batch_size = IntParam("batch_size", batch_values^)

    fn __copyinit__(out self, existing: Self):
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.buffer_size = existing.buffer_size
        self.batch_size = existing.batch_size

    fn __moveinit__(out self, deinit existing: Self):
        self.learning_rate = existing.learning_rate^
        self.discount_factor = existing.discount_factor^
        self.epsilon = existing.epsilon^
        self.epsilon_decay = existing.epsilon_decay^
        self.epsilon_min = existing.epsilon_min^
        self.buffer_size = existing.buffer_size^
        self.batch_size = existing.batch_size^

    fn sample_random(self) -> ReplayHyperparams:
        var buffer = self.buffer_size.sample_random()
        return ReplayHyperparams(
            learning_rate=self.learning_rate.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            epsilon=self.epsilon.sample_random(),
            epsilon_decay=self.epsilon_decay.sample_random(),
            epsilon_min=self.epsilon_min.sample_random(),
            buffer_size=buffer,
            batch_size=self.batch_size.sample_random(),
            min_buffer_size=buffer // 10,  # 10% of buffer size
        )

    fn get_grid_size(self) -> Int:
        return (
            self.learning_rate.num_values
            * self.discount_factor.num_values
            * self.epsilon.num_values
            * self.epsilon_decay.num_values
            * self.epsilon_min.num_values
            * self.buffer_size.num_values()
            * self.batch_size.num_values()
        )

    fn get_grid_config(self, index: Int) -> ReplayHyperparams:
        var remaining = index
        var lr_idx = remaining % self.learning_rate.num_values
        remaining //= self.learning_rate.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var eps_idx = remaining % self.epsilon.num_values
        remaining //= self.epsilon.num_values
        var decay_idx = remaining % self.epsilon_decay.num_values
        remaining //= self.epsilon_decay.num_values
        var min_idx = remaining % self.epsilon_min.num_values
        remaining //= self.epsilon_min.num_values
        var buffer_idx = remaining % self.buffer_size.num_values()
        remaining //= self.buffer_size.num_values()
        var batch_idx = remaining % self.batch_size.num_values()

        var buffer = self.buffer_size.get_value_at_index(buffer_idx)
        return ReplayHyperparams(
            learning_rate=self.learning_rate.get_value_at_index(lr_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            epsilon=self.epsilon.get_value_at_index(eps_idx),
            epsilon_decay=self.epsilon_decay.get_value_at_index(decay_idx),
            epsilon_min=self.epsilon_min.get_value_at_index(min_idx),
            buffer_size=buffer,
            batch_size=self.batch_size.get_value_at_index(batch_idx),
            min_buffer_size=buffer // 10,
        )


# ============================================================================
# Policy Gradient Hyperparameters
# ============================================================================


struct PolicyGradientHyperparams(Copyable, Movable):
    """Hyperparameters for policy gradient agents (REINFORCE, Actor-Critic)."""

    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var entropy_coef: Float64
    var use_baseline: Bool
    var baseline_lr: Float64

    fn __init__(
        out self,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        entropy_coef: Float64 = 0.01,
        use_baseline: Bool = True,
        baseline_lr: Float64 = 0.01,
    ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.baseline_lr = baseline_lr

    fn __copyinit__(out self, existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef
        self.use_baseline = existing.use_baseline
        self.baseline_lr = existing.baseline_lr

    fn __moveinit__(out self, deinit existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef
        self.use_baseline = existing.use_baseline
        self.baseline_lr = existing.baseline_lr

    fn to_csv_header(self) -> String:
        return "actor_lr,critic_lr,discount_factor,entropy_coef,use_baseline,baseline_lr"

    fn to_csv_row(self) -> String:
        return (
            String(self.actor_lr)
            + ","
            + String(self.critic_lr)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.entropy_coef)
            + ","
            + String(self.use_baseline)
            + ","
            + String(self.baseline_lr)
        )


struct PolicyGradientParamSpace(Copyable, Movable):
    """Search space for policy gradient agents."""

    var actor_lr: FloatParam
    var critic_lr: FloatParam
    var discount_factor: FloatParam
    var entropy_coef: FloatParam

    fn __init__(out self):
        self.actor_lr = FloatParam(
            "actor_lr", 0.0001, 0.01, log_scale=True, num_values=4
        )
        self.critic_lr = FloatParam(
            "critic_lr", 0.001, 0.1, log_scale=True, num_values=4
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.9, 0.999, log_scale=False, num_values=3
        )
        self.entropy_coef = FloatParam(
            "entropy_coef", 0.001, 0.1, log_scale=True, num_values=3
        )

    fn __copyinit__(out self, existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef

    fn __moveinit__(out self, deinit existing: Self):
        self.actor_lr = existing.actor_lr^
        self.critic_lr = existing.critic_lr^
        self.discount_factor = existing.discount_factor^
        self.entropy_coef = existing.entropy_coef^

    fn sample_random(self) -> PolicyGradientHyperparams:
        return PolicyGradientHyperparams(
            actor_lr=self.actor_lr.sample_random(),
            critic_lr=self.critic_lr.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            entropy_coef=self.entropy_coef.sample_random(),
            use_baseline=True,
            baseline_lr=self.critic_lr.sample_random(),
        )

    fn get_grid_size(self) -> Int:
        return (
            self.actor_lr.num_values
            * self.critic_lr.num_values
            * self.discount_factor.num_values
            * self.entropy_coef.num_values
        )

    fn get_grid_config(self, index: Int) -> PolicyGradientHyperparams:
        var remaining = index
        var actor_idx = remaining % self.actor_lr.num_values
        remaining //= self.actor_lr.num_values
        var critic_idx = remaining % self.critic_lr.num_values
        remaining //= self.critic_lr.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var entropy_idx = remaining % self.entropy_coef.num_values

        var critic_lr = self.critic_lr.get_value_at_index(critic_idx)
        return PolicyGradientHyperparams(
            actor_lr=self.actor_lr.get_value_at_index(actor_idx),
            critic_lr=critic_lr,
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            entropy_coef=self.entropy_coef.get_value_at_index(entropy_idx),
            use_baseline=True,
            baseline_lr=critic_lr,
        )


# ============================================================================
# PPO Hyperparameters
# ============================================================================


struct PPOHyperparams(Copyable, Movable):
    """Hyperparameters for PPO."""

    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var num_epochs: Int
    var normalize_advantages: Bool

    fn __init__(
        out self,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        num_epochs: Int = 4,
        normalize_advantages: Bool = True,
    ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.normalize_advantages = normalize_advantages

    fn __copyinit__(out self, existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.num_epochs = existing.num_epochs
        self.normalize_advantages = existing.normalize_advantages

    fn __moveinit__(out self, deinit existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.num_epochs = existing.num_epochs
        self.normalize_advantages = existing.normalize_advantages

    fn to_csv_header(self) -> String:
        return "actor_lr,critic_lr,discount_factor,gae_lambda,clip_epsilon,entropy_coef,num_epochs"

    fn to_csv_row(self) -> String:
        return (
            String(self.actor_lr)
            + ","
            + String(self.critic_lr)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.gae_lambda)
            + ","
            + String(self.clip_epsilon)
            + ","
            + String(self.entropy_coef)
            + ","
            + String(self.num_epochs)
        )


struct PPOParamSpace(Copyable, Movable):
    """Search space for PPO."""

    var actor_lr: FloatParam
    var critic_lr: FloatParam
    var discount_factor: FloatParam
    var gae_lambda: FloatParam
    var clip_epsilon: FloatParam
    var entropy_coef: FloatParam
    var num_epochs: IntParam

    fn __init__(out self):
        self.actor_lr = FloatParam(
            "actor_lr", 0.0001, 0.001, log_scale=True, num_values=3
        )
        self.critic_lr = FloatParam(
            "critic_lr", 0.0005, 0.005, log_scale=True, num_values=3
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.95, 0.999, log_scale=False, num_values=3
        )
        self.gae_lambda = FloatParam(
            "gae_lambda", 0.9, 0.99, log_scale=False, num_values=3
        )
        self.clip_epsilon = FloatParam(
            "clip_epsilon", 0.1, 0.3, log_scale=False, num_values=3
        )
        self.entropy_coef = FloatParam(
            "entropy_coef", 0.001, 0.05, log_scale=True, num_values=3
        )
        var epoch_values = List[Int]()
        epoch_values.append(2)
        epoch_values.append(4)
        epoch_values.append(8)
        self.num_epochs = IntParam("num_epochs", epoch_values^)

    fn __copyinit__(out self, existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.num_epochs = existing.num_epochs

    fn __moveinit__(out self, deinit existing: Self):
        self.actor_lr = existing.actor_lr^
        self.critic_lr = existing.critic_lr^
        self.discount_factor = existing.discount_factor^
        self.gae_lambda = existing.gae_lambda^
        self.clip_epsilon = existing.clip_epsilon^
        self.entropy_coef = existing.entropy_coef^
        self.num_epochs = existing.num_epochs^

    fn sample_random(self) -> PPOHyperparams:
        return PPOHyperparams(
            actor_lr=self.actor_lr.sample_random(),
            critic_lr=self.critic_lr.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            gae_lambda=self.gae_lambda.sample_random(),
            clip_epsilon=self.clip_epsilon.sample_random(),
            entropy_coef=self.entropy_coef.sample_random(),
            num_epochs=self.num_epochs.sample_random(),
            normalize_advantages=True,
        )

    fn get_grid_size(self) -> Int:
        return (
            self.actor_lr.num_values
            * self.critic_lr.num_values
            * self.discount_factor.num_values
            * self.gae_lambda.num_values
            * self.clip_epsilon.num_values
            * self.entropy_coef.num_values
            * self.num_epochs.num_values()
        )

    fn get_grid_config(self, index: Int) -> PPOHyperparams:
        var remaining = index
        var actor_idx = remaining % self.actor_lr.num_values
        remaining //= self.actor_lr.num_values
        var critic_idx = remaining % self.critic_lr.num_values
        remaining //= self.critic_lr.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var gae_idx = remaining % self.gae_lambda.num_values
        remaining //= self.gae_lambda.num_values
        var clip_idx = remaining % self.clip_epsilon.num_values
        remaining //= self.clip_epsilon.num_values
        var entropy_idx = remaining % self.entropy_coef.num_values
        remaining //= self.entropy_coef.num_values
        var epochs_idx = remaining % self.num_epochs.num_values()

        return PPOHyperparams(
            actor_lr=self.actor_lr.get_value_at_index(actor_idx),
            critic_lr=self.critic_lr.get_value_at_index(critic_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            gae_lambda=self.gae_lambda.get_value_at_index(gae_idx),
            clip_epsilon=self.clip_epsilon.get_value_at_index(clip_idx),
            entropy_coef=self.entropy_coef.get_value_at_index(entropy_idx),
            num_epochs=self.num_epochs.get_value_at_index(epochs_idx),
            normalize_advantages=True,
        )


# ============================================================================
# Continuous Control Hyperparameters (DDPG, TD3, SAC)
# ============================================================================


struct ContinuousHyperparams(Copyable, Movable):
    """Hyperparameters for continuous control agents (DDPG, TD3, SAC)."""

    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var tau: Float64
    var noise_std: Float64
    var buffer_size: Int
    var batch_size: Int
    # TD3-specific
    var policy_delay: Int
    var target_noise_std: Float64
    var target_noise_clip: Float64
    # SAC-specific
    var alpha: Float64
    var auto_alpha: Bool

    fn __init__(
        out self,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        tau: Float64 = 0.005,
        noise_std: Float64 = 0.1,
        buffer_size: Int = 100000,
        batch_size: Int = 64,
        policy_delay: Int = 2,
        target_noise_std: Float64 = 0.2,
        target_noise_clip: Float64 = 0.5,
        alpha: Float64 = 0.2,
        auto_alpha: Bool = True,
    ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.alpha = alpha
        self.auto_alpha = auto_alpha

    fn __copyinit__(out self, existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.buffer_size = existing.buffer_size
        self.batch_size = existing.batch_size
        self.policy_delay = existing.policy_delay
        self.target_noise_std = existing.target_noise_std
        self.target_noise_clip = existing.target_noise_clip
        self.alpha = existing.alpha
        self.auto_alpha = existing.auto_alpha

    fn __moveinit__(out self, deinit existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.buffer_size = existing.buffer_size
        self.batch_size = existing.batch_size
        self.policy_delay = existing.policy_delay
        self.target_noise_std = existing.target_noise_std
        self.target_noise_clip = existing.target_noise_clip
        self.alpha = existing.alpha
        self.auto_alpha = existing.auto_alpha

    fn to_csv_header(self) -> String:
        return "actor_lr,critic_lr,discount_factor,tau,noise_std,buffer_size,batch_size,policy_delay,target_noise_std,target_noise_clip,alpha,auto_alpha"

    fn to_csv_row(self) -> String:
        return (
            String(self.actor_lr)
            + ","
            + String(self.critic_lr)
            + ","
            + String(self.discount_factor)
            + ","
            + String(self.tau)
            + ","
            + String(self.noise_std)
            + ","
            + String(self.buffer_size)
            + ","
            + String(self.batch_size)
            + ","
            + String(self.policy_delay)
            + ","
            + String(self.target_noise_std)
            + ","
            + String(self.target_noise_clip)
            + ","
            + String(self.alpha)
            + ","
            + String(self.auto_alpha)
        )


struct ContinuousParamSpace(Copyable, Movable):
    """Search space for continuous control agents."""

    var actor_lr: FloatParam
    var critic_lr: FloatParam
    var discount_factor: FloatParam
    var tau: FloatParam
    var noise_std: FloatParam
    var batch_size: IntParam

    fn __init__(out self):
        self.actor_lr = FloatParam(
            "actor_lr", 0.0001, 0.003, log_scale=True, num_values=4
        )
        self.critic_lr = FloatParam(
            "critic_lr", 0.0001, 0.003, log_scale=True, num_values=4
        )
        self.discount_factor = FloatParam(
            "discount_factor", 0.95, 0.999, log_scale=False, num_values=3
        )
        self.tau = FloatParam("tau", 0.001, 0.01, log_scale=True, num_values=3)
        self.noise_std = FloatParam(
            "noise_std", 0.05, 0.3, log_scale=False, num_values=3
        )
        var batch_values = List[Int]()
        batch_values.append(32)
        batch_values.append(64)
        batch_values.append(128)
        batch_values.append(256)
        self.batch_size = IntParam("batch_size", batch_values^)

    fn __copyinit__(out self, existing: Self):
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.batch_size = existing.batch_size

    fn __moveinit__(out self, deinit existing: Self):
        self.actor_lr = existing.actor_lr^
        self.critic_lr = existing.critic_lr^
        self.discount_factor = existing.discount_factor^
        self.tau = existing.tau^
        self.noise_std = existing.noise_std^
        self.batch_size = existing.batch_size^

    fn sample_random(self) -> ContinuousHyperparams:
        return ContinuousHyperparams(
            actor_lr=self.actor_lr.sample_random(),
            critic_lr=self.critic_lr.sample_random(),
            discount_factor=self.discount_factor.sample_random(),
            tau=self.tau.sample_random(),
            noise_std=self.noise_std.sample_random(),
            buffer_size=100000,
            batch_size=self.batch_size.sample_random(),
        )

    fn get_grid_size(self) -> Int:
        return (
            self.actor_lr.num_values
            * self.critic_lr.num_values
            * self.discount_factor.num_values
            * self.tau.num_values
            * self.noise_std.num_values
            * self.batch_size.num_values()
        )

    fn get_grid_config(self, index: Int) -> ContinuousHyperparams:
        var remaining = index
        var actor_idx = remaining % self.actor_lr.num_values
        remaining //= self.actor_lr.num_values
        var critic_idx = remaining % self.critic_lr.num_values
        remaining //= self.critic_lr.num_values
        var df_idx = remaining % self.discount_factor.num_values
        remaining //= self.discount_factor.num_values
        var tau_idx = remaining % self.tau.num_values
        remaining //= self.tau.num_values
        var noise_idx = remaining % self.noise_std.num_values
        remaining //= self.noise_std.num_values
        var batch_idx = remaining % self.batch_size.num_values()

        return ContinuousHyperparams(
            actor_lr=self.actor_lr.get_value_at_index(actor_idx),
            critic_lr=self.critic_lr.get_value_at_index(critic_idx),
            discount_factor=self.discount_factor.get_value_at_index(df_idx),
            tau=self.tau.get_value_at_index(tau_idx),
            noise_std=self.noise_std.get_value_at_index(noise_idx),
            buffer_size=100000,
            batch_size=self.batch_size.get_value_at_index(batch_idx),
        )

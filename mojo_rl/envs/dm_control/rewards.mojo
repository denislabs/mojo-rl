"""Soft indicator functions — Mojo port of `dm_control/utils/rewards.py`.

Every task in the dm_control suite shapes its reward with `tolerance()`, so
this must match the reference bit-for-bit in the cases the suite exercises.
Gated by `tests/dm_control/test_rewards_vs_dm_control.mojo`, which diffs a
dense grid of (x, bounds, margin, sigmoid, value_at_margin) against the
reference `rewards.py` run under Python interop.

Reference: references/dm_control-main/dm_control/utils/rewards.py

Design deltas from the Python original, both deliberate:

- `sigmoid` and `value_at_margin` are COMPTIME parameters. Every call site in
  the suite passes literals for both, so this makes the sigmoid choice a
  compile-time dispatch (no string compare in the reward loop) and turns the
  reference's `ValueError` argument validation into `comptime assert`.
- `bounds` is passed as two scalars (`lower`, `upper`) rather than a tuple.
  Use `inf[DType.float64]()` for a half-open interval, matching the
  reference's `float('inf')`.

All arithmetic is Float64 regardless of the env dtype: the reference is
float64 and the reward is compared against it directly. Callers building a
`Scalar[DTYPE]` reward wrap the result.
"""

from std.math import sqrt, exp, log, cos, cosh, tanh, acos, acosh, atanh, pi


# The value returned by tolerance() at `margin` distance from `bounds`.
comptime DEFAULT_VALUE_AT_MARGIN: Float64 = 0.1

# Sigmoid names (mirror the reference's string literals).
comptime SIGMOID_GAUSSIAN: StaticString = "gaussian"
comptime SIGMOID_HYPERBOLIC: StaticString = "hyperbolic"
comptime SIGMOID_LONG_TAIL: StaticString = "long_tail"
comptime SIGMOID_RECIPROCAL: StaticString = "reciprocal"
comptime SIGMOID_COSINE: StaticString = "cosine"
comptime SIGMOID_LINEAR: StaticString = "linear"
comptime SIGMOID_QUADRATIC: StaticString = "quadratic"
comptime SIGMOID_TANH_SQUARED: StaticString = "tanh_squared"


def _is_known_sigmoid(name: StaticString) -> Bool:
    return (
        name == SIGMOID_GAUSSIAN
        or name == SIGMOID_HYPERBOLIC
        or name == SIGMOID_LONG_TAIL
        or name == SIGMOID_RECIPROCAL
        or name == SIGMOID_COSINE
        or name == SIGMOID_LINEAR
        or name == SIGMOID_QUADRATIC
        or name == SIGMOID_TANH_SQUARED
    )


def _allows_zero_value_at_1(name: StaticString) -> Bool:
    """cosine/linear/quadratic accept value_at_1 == 0; the rest need > 0."""
    return (
        name == SIGMOID_COSINE
        or name == SIGMOID_LINEAR
        or name == SIGMOID_QUADRATIC
    )


@always_inline
def sigmoids[
    sigmoid: StaticString = SIGMOID_GAUSSIAN,
    value_at_1: Float64 = DEFAULT_VALUE_AT_MARGIN,
](x: Float64) -> Float64:
    """Returns 1 when `x` == 0, and falls off to `value_at_1` at |x| == 1.

    Port of `rewards._sigmoids`. `x` is the out-of-bounds distance already
    divided by `margin`, so callers pass x >= 0; the bounded sigmoids
    (cosine/linear/quadratic) rely on that.
    """
    comptime assert _is_known_sigmoid(sigmoid), "Unknown sigmoid type."
    comptime if _allows_zero_value_at_1(sigmoid):
        comptime assert (
            value_at_1 >= 0.0 and value_at_1 < 1.0
        ), "`value_at_1` must be nonnegative and < 1 for this sigmoid."
    else:
        comptime assert (
            value_at_1 > 0.0 and value_at_1 < 1.0
        ), "`value_at_1` must be strictly between 0 and 1 for this sigmoid."

    comptime if sigmoid == SIGMOID_GAUSSIAN:
        var scale = sqrt(-2.0 * log(value_at_1))
        return exp(-0.5 * (x * scale) * (x * scale))

    elif sigmoid == SIGMOID_HYPERBOLIC:
        var scale = acosh(1.0 / value_at_1)
        return 1.0 / cosh(x * scale)

    elif sigmoid == SIGMOID_LONG_TAIL:
        var scale = sqrt(1.0 / value_at_1 - 1.0)
        return 1.0 / ((x * scale) * (x * scale) + 1.0)

    elif sigmoid == SIGMOID_RECIPROCAL:
        var scale = 1.0 / value_at_1 - 1.0
        return 1.0 / (abs(x) * scale + 1.0)

    elif sigmoid == SIGMOID_COSINE:
        var scale = acos(2.0 * value_at_1 - 1.0) / pi
        var scaled_x = x * scale
        if abs(scaled_x) < 1.0:
            return (1.0 + cos(pi * scaled_x)) / 2.0
        return 0.0

    elif sigmoid == SIGMOID_LINEAR:
        var scale = 1.0 - value_at_1
        var scaled_x = x * scale
        if abs(scaled_x) < 1.0:
            return 1.0 - scaled_x
        return 0.0

    elif sigmoid == SIGMOID_QUADRATIC:
        var scale = sqrt(1.0 - value_at_1)
        var scaled_x = x * scale
        if abs(scaled_x) < 1.0:
            return 1.0 - scaled_x * scaled_x
        return 0.0

    else:  # SIGMOID_TANH_SQUARED
        var scale = atanh(sqrt(1.0 - value_at_1))
        var t = tanh(x * scale)
        return 1.0 - t * t


@always_inline
def tolerance[
    sigmoid: StaticString = SIGMOID_GAUSSIAN,
    value_at_margin: Float64 = DEFAULT_VALUE_AT_MARGIN,
](
    x: Float64,
    lower: Float64 = 0.0,
    upper: Float64 = 0.0,
    margin: Float64 = 0.0,
) -> Float64:
    """Returns 1 when `x` is inside `[lower, upper]`, and 0..1 outside.

    Port of `rewards.tolerance`. `margin == 0` gives a hard indicator; a
    positive `margin` decays with `sigmoid` so that the value is
    `value_at_margin` at exactly `margin` beyond the nearest bound.

    The reference validates `lower <= upper` and `margin >= 0` at runtime and
    raises `ValueError`; both are caller bugs rather than data, so this
    returns without a `raises` and leaves the check to debug_assert-free
    callers — the parity test covers the valid domain.
    """
    var in_bounds = lower <= x and x <= upper
    if margin == 0.0:
        return 1.0 if in_bounds else 0.0
    if in_bounds:
        return 1.0
    # Distance past the nearest bound, normalized by margin (always >= 0).
    var d = (lower - x) / margin if x < lower else (x - upper) / margin
    return sigmoids[sigmoid, value_at_margin](d)

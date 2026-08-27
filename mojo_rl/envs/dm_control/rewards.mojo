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

`DTYPE` defaults to `float64` and is INFERRED from `x`, so every CPU call site
reads exactly as before and keeps float64 arithmetic — the reference is float64
and the CPU parity gates diff against it directly.

⚠ The `DTYPE` parameter exists for the GPU hooks, and it is not cosmetic:
**Metal has no `double`**, so a kernel that calls a `Float64 -> Float64` helper
is rejected outright ("returns unsupported type 'double'") — see the standing
warning at `physics3d/solver/newton_solve.mojo:1046`. GPU reward hooks must
instantiate this at the env dtype.

⚠ The per-sigmoid `scale` is a function of the COMPTIME `value_at_1` alone, so
it looks like it should be a `comptime` Float64 constant cast once. It cannot
be: Mojo's compile-time interpreter cannot fold `acos`/`log`/`atanh`
("LLVM could not constant fold intrinsic: llvm.acos"). It is therefore computed
at runtime in `DTYPE` like everything else, which means the float32
instantiation carries a float32 `scale`. That costs ~1e-7 relative and is what
`test_rewards_vs_dm_control.test_tolerance_float32_matches_float64` bounds.
"""

from std.math import sqrt, exp, log, cos, cosh, tanh, acos, acosh, atanh, pi

from mojo_rl.envs.dm_control.dtype_math import log_accurate


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
    DTYPE: DType = DType.float64,
](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """Returns 1 when `x` == 0, and falls off to `value_at_1` at |x| == 1.

    Port of `rewards._sigmoids`. `x` is the out-of-bounds distance already
    divided by `margin`, so callers pass x >= 0; the bounded sigmoids
    (cosine/linear/quadratic) rely on that.

    `DTYPE` is inferred from `x`; see the module docstring for why the GPU
    path must not leave it at the float64 default.

    ⚠ WHY THIS DISPATCHES INSTEAD OF JUST CARRYING THE BODY. `std.math`'s
    `cos`/`cosh`/`tanh`/`acos`/`acosh`/`atanh` are declared
    `where dtype.is_floating_point()`, and Mojo type-checks a generic body
    eagerly — so the body needs that evidence. Putting the constraint on this
    function instead only moves the problem: the GPU reward hooks are trait
    methods whose `DTYPE` is UNCONSTRAINED, and a trait signature cannot grow
    a `where` clause without every implementing config growing one too (13
    files, for a property that is true of every physics dtype anyway).
    So the constrained body lives in `_sigmoids_impl` and this dispatches to it
    on a COMPTIME-known dtype, where the constraint is trivially provable.
    """
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](
            _sigmoids_impl[sigmoid, value_at_1, DType.float32](
                rebind[Float32](x)
            )
        )
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](
            _sigmoids_impl[sigmoid, value_at_1, DType.float64](
                rebind[Float64](x)
            )
        )
    else:
        comptime assert False, (
            "rewards.sigmoids: only float32 / float64 are supported. Add the"
            " branch here rather than widening the constraint — see the note"
            " above."
        )


@always_inline
def _sigmoids_impl[
    sigmoid: StaticString,
    value_at_1: Float64,
    DTYPE: DType,
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """The single sigmoid body. Reached only through `sigmoids`, which binds
    `DTYPE` to a concrete float type first (see its docstring)."""
    comptime assert _is_known_sigmoid(sigmoid), "Unknown sigmoid type."
    comptime if _allows_zero_value_at_1(sigmoid):
        comptime assert (
            value_at_1 >= 0.0 and value_at_1 < 1.0
        ), "`value_at_1` must be nonnegative and < 1 for this sigmoid."
    else:
        comptime assert (
            value_at_1 > 0.0 and value_at_1 < 1.0
        ), "`value_at_1` must be strictly between 0 and 1 for this sigmoid."

    comptime ONE = Scalar[DTYPE](1.0)
    comptime ZERO = Scalar[DTYPE](0.0)
    comptime TWO = Scalar[DTYPE](2.0)
    comptime V = Scalar[DTYPE](value_at_1)

    comptime if sigmoid == SIGMOID_GAUSSIAN:
        # ⚠⚠ `log_accurate`, NOT `std.math.log`. `log(0.1)` comes back 8.0e-11
        # low, which makes this scale 4.0e-11 low, and the exponent below
        # AMPLIFIES that by u^2/2 — 3.9e-09 relative at u = 9.9, measured on
        # `reach_duplo_features`' reward ramp against a 1e-12 gate. See
        # `dtype_math.log_accurate`.
        #
        # ⚠ `std.math.exp` IS ALSO INEXACT and is deliberately left alone:
        # measured 2.8e-13 relative at exp(-1) growing to 8.2e-12 at exp(-20).
        # That is below every reward gate in this tree and there is no
        # identity to escape to, unlike `log`. Recorded so the next residual
        # here starts from a number rather than a guess.
        var scale = sqrt(Scalar[DTYPE](-2.0) * log_accurate[DTYPE](V))
        return exp(Scalar[DTYPE](-0.5) * (x * scale) * (x * scale))

    elif sigmoid == SIGMOID_HYPERBOLIC:
        var scale = acosh(ONE / V)
        return ONE / cosh(x * scale)

    elif sigmoid == SIGMOID_LONG_TAIL:
        var scale = sqrt(ONE / V - ONE)
        return ONE / ((x * scale) * (x * scale) + ONE)

    elif sigmoid == SIGMOID_RECIPROCAL:
        var scale = ONE / V - ONE
        return ONE / (abs(x) * scale + ONE)

    elif sigmoid == SIGMOID_COSINE:
        var scale = acos(TWO * V - ONE) / Scalar[DTYPE](pi)
        var scaled_x = x * scale
        if abs(scaled_x) < ONE:
            return (ONE + cos(Scalar[DTYPE](pi) * scaled_x)) / TWO
        return ZERO

    elif sigmoid == SIGMOID_LINEAR:
        var scale = ONE - V
        var scaled_x = x * scale
        if abs(scaled_x) < ONE:
            return ONE - scaled_x
        return ZERO

    elif sigmoid == SIGMOID_QUADRATIC:
        var scale = sqrt(ONE - V)
        var scaled_x = x * scale
        if abs(scaled_x) < ONE:
            return ONE - scaled_x * scaled_x
        return ZERO

    else:  # SIGMOID_TANH_SQUARED
        var scale = atanh(sqrt(ONE - V))
        var t = tanh(x * scale)
        return ONE - t * t


@always_inline
def tolerance[
    sigmoid: StaticString = SIGMOID_GAUSSIAN,
    value_at_margin: Float64 = DEFAULT_VALUE_AT_MARGIN,
    DTYPE: DType = DType.float64,
](
    x: Scalar[DTYPE],
    lower: Scalar[DTYPE] = 0.0,
    upper: Scalar[DTYPE] = 0.0,
    margin: Scalar[DTYPE] = 0.0,
) -> Scalar[DTYPE]:
    """Returns 1 when `x` is inside `[lower, upper]`, and 0..1 outside.

    Port of `rewards.tolerance`. `margin == 0` gives a hard indicator; a
    positive `margin` decays with `sigmoid` so that the value is
    `value_at_margin` at exactly `margin` beyond the nearest bound.

    The reference validates `lower <= upper` and `margin >= 0` at runtime and
    raises `ValueError`; both are caller bugs rather than data, so this
    returns without a `raises` and leaves the check to debug_assert-free
    callers — the parity test covers the valid domain.
    """
    comptime ONE = Scalar[DTYPE](1.0)
    comptime ZERO = Scalar[DTYPE](0.0)
    var in_bounds = lower <= x and x <= upper
    if margin == ZERO:
        return ONE if in_bounds else ZERO
    if in_bounds:
        return ONE
    # Distance past the nearest bound, normalized by margin (always >= 0).
    var d = (lower - x) / margin if x < lower else (x - upper) / margin
    return sigmoids[sigmoid, value_at_margin, DTYPE](d)

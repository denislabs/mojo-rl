"""DTYPE-generic math shims callable from a GPU hook.

⚠ WHY THIS MODULE EXISTS. `std.math`'s transcendentals are declared
`where dtype.is_floating_point()`, Mojo type-checks a generic body EAGERLY, and
the GPU hooks are trait methods whose `DTYPE` is UNCONSTRAINED — so a hook that
calls `log(x)` on a `Scalar[DTYPE]` fails with *"invalid call to 'log': lacking
evidence to prove correctness"*. Putting the constraint on the caller only moves
the error, because a trait signature cannot grow a `where` clause without every
implementing config growing one too.

The escape is always the same: dispatch on a COMPTIME-KNOWN dtype, so the
constrained body is instantiated at a concrete float type where the constraint
is trivially provable. This module holds that boilerplate once instead of a
fourth copy — `dm_control/rewards.sigmoids` and
`dm_control/gpu_reset.standard_normal` are the first two, and both predate it.

See `feedback_where_clause_cannot_cross_trait_boundary`. Add shims here as
hooks need them; do NOT widen a trait signature to satisfy `std.math`.
"""

from std.math import log, sin, cos, sqrt, inf, abs


@always_inline
def log1p_dt[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`log(1 + x)`, callable from a GPU hook.

    Spelled `log(1 + x)` rather than a true `log1p` because that is exactly
    what the CPU hooks compute (`np.log1p` on the touch sensors, transcribed as
    `log(1.0 + toe)`), and the two paths are diffed element-wise. A more
    accurate `log1p` here would be a REAL divergence from the gated CPU path
    for small x, not an improvement.

    ⚠ `x` may be NEGATIVE by design: `touch_sphere_site_gpu` returns
    `TOUCH_UNSUPPORTED_ZONE` (-1.0) for a zone type it does not implement, and
    the resulting NaN is the intended loud signal. Do not clamp it away.
    """
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](
            _log1p_impl[DType.float32](rebind[Float32](x))
        )
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](
            _log1p_impl[DType.float64](rebind[Float64](x))
        )
    else:
        comptime assert False, (
            "dtype_math.log1p_dt: only float32 / float64 are supported. Add"
            " the branch here rather than widening a trait signature."
        )


@always_inline
def _log1p_impl[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """The body. Reached only through `log1p_dt`, which binds `DTYPE` to a
    concrete float type first — see the module docstring."""
    return log(Scalar[DTYPE](1.0) + x)


@always_inline
def sin_dt[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`sin(x)`, callable from a GPU hook. See the module docstring."""
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](_sin_impl[DType.float32](rebind[Float32](x)))
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](_sin_impl[DType.float64](rebind[Float64](x)))
    else:
        comptime assert False, "dtype_math.sin_dt: float32 / float64 only."


@always_inline
def cos_dt[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`cos(x)`, callable from a GPU hook. See the module docstring."""
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](_cos_impl[DType.float32](rebind[Float32](x)))
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](_cos_impl[DType.float64](rebind[Float64](x)))
    else:
        comptime assert False, "dtype_math.cos_dt: float32 / float64 only."


@always_inline
def _sin_impl[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    return sin(x)


@always_inline
def _cos_impl[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    return cos(x)


@always_inline
def asinh_dt[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`np.arcsinh`, callable from a GPU hook. See the module docstring.

    Spelled `log(x + sqrt(x*x + 1))` because that is exactly what the CPU
    twin computes (`quadruped_config._asinh`), and the two paths are diffed
    element-wise. `std.math` has no `asinh`, so there is no more accurate
    form available to diverge to.

    ⚠⚠ THE SIGN FOLD IS LOAD-BEARING, NOT TIDINESS. Evaluated directly at
    large NEGATIVE x, `x + sqrt(x*x + 1)` is a catastrophic cancellation: in
    float32, `x*x` for x = -1435 already discards the `+ 1`, and the sum of
    two ~1435 numbers of opposite sign is pure rounding residue. Measured on
    quadruped's four toe force sensors, whose z components were
    -1435.0, -1440.9, -1406.7, -1435.9:

        direct form  ->  -7.9123010635 for ALL FOUR, bit-identical
        stable form  ->  -7.9621, -7.9662, -7.9421, -7.9627

    Four distinct forces collapsing onto one number is not an error bar, it
    is the dim ceasing to carry information — and it agreed with itself
    perfectly, so only a CPU cross-check could see it.

    `asinh` is ODD, so evaluating on |x| and restoring the sign is exact and
    removes the cancellation entirely (`|x| + sqrt(x*x+1)` sums two positives).
    `quadruped_config._asinh` carries the same fold for the same reason —
    float64 merely degrades later, it does not escape.
    """
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](
            _asinh_impl[DType.float32](rebind[Float32](x))
        )
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](
            _asinh_impl[DType.float64](rebind[Float64](x))
        )
    else:
        comptime assert False, "dtype_math.asinh_dt: float32 / float64 only."


@always_inline
def _asinh_impl[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    var a = abs(x)
    var r = log(a + sqrt(a * a + Scalar[DTYPE](1.0)))
    return -r if x < Scalar[DTYPE](0) else r


@always_inline
def sqrt_dt[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`sqrt(x)`, callable from a GPU hook. See the module docstring."""
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](
            _sqrt_impl[DType.float32](rebind[Float32](x))
        )
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](
            _sqrt_impl[DType.float64](rebind[Float64](x))
        )
    else:
        comptime assert False, "dtype_math.sqrt_dt: float32 / float64 only."


@always_inline
def _sqrt_impl[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    return sqrt(x)


@always_inline
def inf_dt[DTYPE: DType]() -> Scalar[DTYPE]:
    """`+inf` in DTYPE, callable from a GPU hook.

    Needed because `rewards.tolerance` takes `upper = inf` to mean "no upper
    bound", and several dm_control rewards are one-sided that way. A large
    FINITE stand-in would be a silent divergence: `tolerance` computes
    `(x - upper) / margin`, so a finite `upper` turns an unbounded-above
    reward into one that decays once x passes the stand-in.
    """
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](_inf_impl[DType.float32]())
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](_inf_impl[DType.float64]())
    else:
        comptime assert False, "dtype_math.inf_dt: float32 / float64 only."


@always_inline
def _inf_impl[DTYPE: DType]() -> Scalar[DTYPE] where DTYPE.is_floating_point():
    return inf[DTYPE]()

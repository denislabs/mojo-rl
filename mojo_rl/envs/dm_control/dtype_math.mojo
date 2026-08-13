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

from std.math import log, sin, cos, sqrt, inf, abs, atanh


@always_inline
def log1p_accurate(x: Float64) -> Float64:
    """`log(1 + x)` to full float64, for a term gated against numpy.

    ⚠⚠ `std.math.log1p` IS NOT ACCURATE ENOUGH TO GATE AGAINST `np.log1p`, and
    neither is `log(1 + x)`. Measured against libm over 54 samples spanning
    1e-12 to 2e3 (both signs), worst RELATIVE error:

        std.math.log1p(x)                     1.99e-08
        log(1+x) * (x / ((1+x) - 1))          1.09e-09    (Kahan; bounded by
                                                           `log`'s own error)
        2 * atanh(x / (2 + x))                1.67e-15

    `std.math.log` is itself only ~1e-10 relative here — `log(10.0)` comes back
    2.6e-10 low — so no formula spelled in terms of it can do better. The
    `atanh` identity avoids `log` entirely and is the only one that reaches
    float64.

    That is not academic. This showed up as a **3.3e-07** disagreement in
    `reach_site_features`' `joints_torque` observable while every input to it
    (`cfrc_int`, `cacc`, `subtree_com`, `site_xpos_acc`, `xquat_acc`, and the
    raw sensor 3-vector) matched MuJoCo to 1e-15. Two wrong causes were filed
    and refuted first — the constraint solver (the residual survives zeroing
    every `frictionloss`, and MuJoCo does not move when tightened to 500
    iterations at 1e-14) and `qacc` (post-`mj_Euler` `qacc` is not
    `mj_forward`'s, a known 1.5% false alarm). The arithmetic was last on the
    list and it was the culprit.

    ⚠ THE IDENTITY INVERTS FOR LARGE x, hence the crossover. As x grows,
    `x / (2 + x)` approaches 1 and `atanh` loses the precision the small-x case
    gains: measured absolute error 1.1e-14 at x=1e4, 1.3e-11 at 1e6, 2.6e-08 at
    1e9, 8e-04 at 1e15, and `inf` from ~1e17 where the argument rounds to
    exactly 1. Above the crossover there is no cancellation left to correct, so
    `log(1 + x)` is both adequate and safe.

    Domain matches `np.log1p`: `-inf` at x = -1, NaN below it.
    """
    if x > 1.0e4:
        return log(1.0 + x)
    return 2.0 * atanh(x / (2.0 + x))


@always_inline
def log_accurate[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """`log(x)` to full float64 over [1e-5, 1e5], for a gated constant.

    ⚠⚠ `std.math.log` IS NOT FLOAT64-ACCURATE, AND IT IS WORST EXACTLY WHERE
    IT IS USED. Measured against `np.log` over a decade sweep:

        x      1e-6    1e-5    1e-3    0.01    0.1     10      100     1e4
        std   4.3e-14 3.4e-11 3.3e-14 1.9e-11 8.0e-11 1.1e-10 5.6e-11 2.0e-11
        this  4.1e-13 8.2e-15 4.9e-15 0.0     0.0     0.0     1.9e-16 1.8e-14

    Its error is ERRATIC rather than monotone — exact at 0.5 and 2.0, 1.1e-10
    at 10 — so no bound can be inferred from a single spot check.

    `2*atanh((x-1)/(x+1))` is the same identity `log1p_accurate` uses, and it
    reaches float64 for the same reason: it never calls `log`. It INVERTS
    outside [1e-5, 1e5], where `(x-1)/(x+1)` approaches +-1 and `atanh` loses
    what the identity gains (4.1e-13 at 1e-6, 6.7e-12 at 1e7) — and there
    `std.math.log` happens to be at its best, so the fallback is not a
    compromise.

    ⚠ THIS IS NOT ACADEMIC EITHER. `rewards.sigmoids`' gaussian scale is
    `sqrt(-2 log(value_at_1))`, and `log(0.1)` being 8.0e-11 low made the
    scale 4.0e-11 low — which the exponent AMPLIFIES BY u^2/2. On
    `reach_duplo_features`' reward ramp that reached **3.9e-09 relative** at 28
    cm, against a gate written at 1e-12. See `log1p_accurate` for the same
    lesson found through `joints_torque`.

    ⚠ THE CALLER MUST CARRY THE FLOATING-POINT EVIDENCE. Unlike the `*_dt`
    shims below this is a plain constrained generic, so it is NOT callable from
    an unconstrained trait method — see the module docstring. Add a dispatching
    `log_dt` if a GPU hook ever needs it.
    """
    comptime LO = Scalar[DTYPE](1.0e-5)
    comptime HI = Scalar[DTYPE](1.0e5)
    if x < LO or x > HI:
        return log(x)
    return Scalar[DTYPE](2.0) * atanh(
        (x - Scalar[DTYPE](1.0)) / (x + Scalar[DTYPE](1.0))
    )


@always_inline
def log1p_dt[DTYPE: DType](x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`log(1 + x)`, callable from a GPU hook.

    Spelled `log(1 + x)` rather than a true `log1p` because that is exactly
    what the CPU hooks compute (`np.log1p` on the touch sensors, transcribed as
    `log(1.0 + toe)`), and the two paths are diffed element-wise. A more
    accurate `log1p` here would be a REAL divergence from the gated CPU path
    for small x, not an improvement.

    ⚠ DELIBERATELY NOT `log1p_accurate` ABOVE, and this is the one place the
    distinction is a choice rather than an oversight. The touch observables in
    `manipulator` / `dog` are transcribed as `log(1.0 + x)` on BOTH paths and
    gated as such; swapping one side would move numbers that are currently
    green for a reason unrelated to those ports. Making them accurate is worth
    doing, with its own before/after — not as a side effect of a shared helper.

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

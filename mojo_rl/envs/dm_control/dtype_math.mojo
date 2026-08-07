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

from std.math import log


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

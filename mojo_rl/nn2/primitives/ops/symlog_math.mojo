"""Canonical symlog / symexp math — single source of truth (audit L4).

Before this module the symlog formulas lived in two places that could
drift independently:
  * `loss/two_hot.mojo` — free scalar `symlog` / `symexp` helpers
  * `primitives/ops/symlog_op.mojo` — `SymlogOp` ElementOp forward/backward

Both now delegate here. The scalar and SIMD forms are mathematically
identical to the previous in-line code (negation is exact in IEEE-754,
so the branch form `-log(1 - x)` and the `sign · log(1 + |x|)` form
agree bit-for-bit), so this dedup is bit-identical — guarded by the
existing `test_elementwise_symlog_parity.mojo` and two_hot tests.
"""

from std.math import exp, log
from std.math import abs as math_abs

from ...constants import DT


@always_inline
def symlog(x: Scalar[DT]) -> Scalar[DT]:
    """Symmetric log: `sign(x) * ln(1 + |x|)`."""
    var abs_x = x if x >= Scalar[DT](0) else -x
    var sgn: Scalar[DT] = 1 if x >= Scalar[DT](0) else -1
    return sgn * log(Scalar[DT](1) + abs_x)


@always_inline
def symexp(x: Scalar[DT]) -> Scalar[DT]:
    """Inverse of symlog: `sign(x) * (exp(|x|) - 1)`."""
    var abs_x = x if x >= Scalar[DT](0) else -x
    var sgn: Scalar[DT] = 1 if x >= Scalar[DT](0) else -1
    return sgn * (exp(abs_x) - Scalar[DT](1))


@always_inline
def symlog_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
    """SIMD symlog."""
    var abs_x = math_abs(x)
    var sgn = x.ge(SIMD[DT, W](0)).select(SIMD[DT, W](1), SIMD[DT, W](-1))
    return sgn * log(SIMD[DT, W](1) + abs_x)

"""Phase 0 planners: ValueEncoding trait promotion.

Verifies the trait values land at the new home AND remain reachable through
the muzero/strategies.mojo re-export shim (so existing imports keep working).

Usage:
    pixi run mojo run -I . tests/planners/common/test_value_encoding.mojo
"""

from std.testing import assert_equal, assert_true, assert_false

from mojo_rl.planners.common import (
    CategoricalEncoding,
    ScalarEncoding,
    SymlogEncoding,
)
# Source-compat shim:
from mojo_rl.deep_agents.muzero.strategies import (
    CategoricalEncoding as ShimCategorical,
    ScalarEncoding as ShimScalar,
    SymlogEncoding as ShimSymlog,
)


def test_new_home_values() raises:
    assert_true(CategoricalEncoding.IS_DISTRIBUTIONAL)
    assert_true(CategoricalEncoding.USE_SCALAR_TRANSFORM)
    assert_false(ScalarEncoding.IS_DISTRIBUTIONAL)
    assert_false(ScalarEncoding.USE_SCALAR_TRANSFORM)
    assert_false(SymlogEncoding.IS_DISTRIBUTIONAL)
    assert_true(SymlogEncoding.USE_SCALAR_TRANSFORM)


def test_shim_reaches_same_constants() raises:
    # Re-export must surface the *same* comptime values as the canonical home.
    assert_equal(
        ShimCategorical.IS_DISTRIBUTIONAL,
        CategoricalEncoding.IS_DISTRIBUTIONAL,
    )
    assert_equal(
        ShimCategorical.USE_SCALAR_TRANSFORM,
        CategoricalEncoding.USE_SCALAR_TRANSFORM,
    )
    assert_equal(
        ShimScalar.IS_DISTRIBUTIONAL, ScalarEncoding.IS_DISTRIBUTIONAL
    )
    assert_equal(
        ShimSymlog.USE_SCALAR_TRANSFORM, SymlogEncoding.USE_SCALAR_TRANSFORM
    )


def main() raises:
    print("=== Phase 0 planners: value_encoding ===")
    test_new_home_values()
    print("  PASS new home values")
    test_shim_reaches_same_constants()
    print("  PASS shim re-export parity")
    print("OK")

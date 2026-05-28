"""ValueEncoding trait — canonical-home comptime constants.

Was originally a shim-parity test (verifying ``muzero.strategies``
re-exported value_encoding correctly). The shim was retired
2026-05-21 once all consumers migrated to import directly from
``mojo_rl.planners.common.value_encoding``. The shim-parity half of
this test was dropped at the same time; the remaining test just
locks in the canonical comptime values so an accidental flip surfaces.

Usage:
    pixi run mojo run -I . tests/planners/common/test_value_encoding.mojo
"""

from std.testing import assert_true, assert_false

from mojo_rl.planners.common import (
    CategoricalEncoding,
    ScalarEncoding,
    SymlogEncoding,
)


def test_new_home_values() raises:
    assert_true(CategoricalEncoding.IS_DISTRIBUTIONAL)
    assert_true(CategoricalEncoding.USE_SCALAR_TRANSFORM)
    assert_false(ScalarEncoding.IS_DISTRIBUTIONAL)
    assert_false(ScalarEncoding.USE_SCALAR_TRANSFORM)
    assert_false(SymlogEncoding.IS_DISTRIBUTIONAL)
    assert_true(SymlogEncoding.USE_SCALAR_TRANSFORM)


def main() raises:
    print("=== planners.common.value_encoding ===")
    test_new_home_values()
    print("  PASS new home values")
    print("OK")

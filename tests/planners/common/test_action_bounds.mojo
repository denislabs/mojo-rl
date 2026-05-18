"""Phase 0 planners: action bounds helpers (clip / tanh_squash / scale_to_range).

Usage:
    pixi run mojo run -I . tests/planners/common/test_action_bounds.mojo
"""

from std.math import abs as math_abs, tanh as math_tanh
from std.testing import assert_equal, assert_true

from mojo_rl.planners.common import (
    clip, tanh_squash, scale_to_range, clip_inplace,
)


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-12) -> Bool:
    return math_abs(a - b) <= tol


def test_clip() raises:
    assert_equal(clip(0.5, 0.0, 1.0), 0.5)
    assert_equal(clip(-1.0, 0.0, 1.0), 0.0)
    assert_equal(clip(2.0, 0.0, 1.0), 1.0)
    assert_equal(clip(0.0, 0.0, 1.0), 0.0)
    assert_equal(clip(1.0, 0.0, 1.0), 1.0)
    # Symmetric bounds (MPPI-style).
    assert_equal(clip(1.5, -1.0, 1.0), 1.0)
    assert_equal(clip(-1.5, -1.0, 1.0), -1.0)


def test_tanh_squash_matches_stdlib() raises:
    for i in range(-5, 6):
        var x = Float64(i) * 0.7
        assert_true(_approx(tanh_squash(x), math_tanh(x)))


def test_scale_to_range_round_trip() raises:
    # x in [-1, 1] should map linearly into [lo, hi].
    assert_true(_approx(scale_to_range(-1.0, 2.0, 8.0), 2.0))
    assert_true(_approx(scale_to_range(1.0, 2.0, 8.0), 8.0))
    assert_true(_approx(scale_to_range(0.0, 2.0, 8.0), 5.0))
    # Negative range still works.
    assert_true(_approx(scale_to_range(0.5, -10.0, 10.0), 5.0))


def test_clip_inplace() raises:
    var buf: List[Float64] = [-2.0, -0.5, 0.0, 0.7, 3.0]
    clip_inplace(buf, -1.0, 1.0)
    assert_equal(buf[0], -1.0)
    assert_equal(buf[1], -0.5)
    assert_equal(buf[2], 0.0)
    assert_equal(buf[3], 0.7)
    assert_equal(buf[4], 1.0)


def main() raises:
    print("=== Phase 0 planners: action_bounds ===")
    test_clip()
    print("  PASS clip")
    test_tanh_squash_matches_stdlib()
    print("  PASS tanh_squash matches stdlib")
    test_scale_to_range_round_trip()
    print("  PASS scale_to_range")
    test_clip_inplace()
    print("  PASS clip_inplace")
    print("OK")

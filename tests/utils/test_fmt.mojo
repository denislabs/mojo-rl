# +--------------------------------------------------------------------------+ #
# | Fixed-width formatting — the short-value cases that crashed a live arm
# +--------------------------------------------------------------------------+ #
"""Regression gate for `mojo_rl/utils/fmt.mojo`.

Every case below is a value whose FORMATTED WIDTH is short. That is the whole
point: `String(x)[byte=0:6]` aborted a running SO-101 teleop loop mid-motion
because one joint angle printed as `3.29`, and the follower was left holding
torque (an abort does not run `finally`). Any formatter that reaches into a
control loop has to be total over its input.

Run: pixi run mojo run -I . tests/utils/test_fmt.mojo
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.utils.fmt import col, fixed, pad_left, pad_right


def test_short_values_do_not_truncate_or_crash() raises:
    """The exact shapes the teleop loop printed when it died."""
    assert_equal(fixed(3.2967, 4), String("3.2967"))
    assert_equal(fixed(-0.043, 4), String("-0.0430"))
    assert_equal(fixed(0.0, 4), String("0.0000"))
    assert_equal(fixed(3.29, 4), String("3.2900"))
    # A one-digit value in a six-wide column: the case that asserted.
    assert_equal(col(3.29, 8, 4), String("  3.2900"))


def test_rounding_is_half_away_from_zero() raises:
    assert_equal(fixed(0.125, 2), String("0.13"))
    assert_equal(fixed(-0.125, 2), String("-0.13"))
    assert_equal(fixed(1.005, 1), String("1.0"))
    assert_equal(fixed(-1.05, 1), String("-1.1"))


def test_carry_across_the_decimal_point() raises:
    """`0.999 -> 1.000`, not `0.1000` — the integer/fraction split has to be
    taken AFTER rounding, which is where a naive implementation breaks."""
    assert_equal(fixed(0.999, 2), String("1.00"))
    assert_equal(fixed(-0.999, 2), String("-1.00"))
    assert_equal(fixed(9.99, 1), String("10.0"))


def test_zero_padding_of_the_fraction() raises:
    """`1.05` must not print as `1.5`."""
    assert_equal(fixed(1.05, 2), String("1.05"))
    assert_equal(fixed(1.005, 3), String("1.005"))
    assert_equal(fixed(-2.001, 3), String("-2.001"))


def test_negative_zero_and_tiny_magnitudes() raises:
    """A value that rounds to zero from below must not print `-0.00`'s sign
    inconsistently with its neighbours in a column."""
    assert_equal(fixed(-0.0001, 2), String("0.00"))
    assert_equal(fixed(0.0001, 2), String("0.00"))


def test_decimals_zero_is_an_integer() raises:
    assert_equal(fixed(3.7, 0), String("4"))
    assert_equal(fixed(-3.7, 0), String("-4"))
    assert_equal(fixed(3.2, 0), String("3"))


def test_non_finite_values_are_short_words() raises:
    """NaN reaching a log line should print `nan`, not abort and not a wall of
    digits — a control loop's telemetry must survive its own bad numbers."""
    var nan = Float64(0.0) / Float64(0.0)
    var inf = Float64(1.0) / Float64(0.0)
    assert_equal(fixed(nan, 3), String("nan"))
    assert_equal(fixed(inf, 3), String("inf"))
    assert_equal(fixed(-inf, 3), String("-inf"))
    # And they still fit a column without asserting.
    assert_equal(col(nan, 6, 3), String("   nan"))


def test_padding_never_drops_digits() raises:
    """An over-wide value keeps every digit and pushes the row out. Silently
    truncating a number to fit a column is how a wrong reading looks right."""
    assert_equal(pad_left(String("12345678"), 4), String("12345678"))
    assert_equal(pad_right(String("12345678"), 4), String("12345678"))
    assert_equal(pad_left(String("7"), 4), String("   7"))
    assert_equal(pad_right(String("7"), 4), String("7   "))
    assert_true(
        col(-123456.789, 6, 3).byte_length() >= 11, "wide value survives"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

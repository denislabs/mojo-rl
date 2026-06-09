"""Unit tests for the zero-series two-player zero-sum sign conventions.

Guards the legacy MuZero P0 bug (missing two-player value-target sign flip) by
checking the sign helpers against a hand-computed TicTacToe trajectory.

Run:
    pixi run mojo run -I . tests/deep_agents2/test_zero_signs.mojo
"""

from std.testing import assert_equal, TestSuite

from mojo_rl.deep_agents2.zero.signs import (
    az_value_target,
    zero_sum_sign,
    flip_for_perspective,
    RESULT_P0_WINS,
    RESULT_P1_WINS,
    RESULT_DRAW,
)


def test_az_value_target_p0_wins() raises:
    assert_equal(az_value_target(RESULT_P0_WINS, 0), 1.0)   # winner's view
    assert_equal(az_value_target(RESULT_P0_WINS, 1), -1.0)  # loser's view


def test_az_value_target_p1_wins() raises:
    assert_equal(az_value_target(RESULT_P1_WINS, 1), 1.0)
    assert_equal(az_value_target(RESULT_P1_WINS, 0), -1.0)


def test_az_value_target_draw() raises:
    assert_equal(az_value_target(RESULT_DRAW, 0), 0.0)
    assert_equal(az_value_target(RESULT_DRAW, 1), 0.0)


def test_trajectory_p0_wins() raises:
    # TicTacToe: P0 moves at even t, P1 at odd t; P0 wins the game.
    # Per-step value targets must alternate +1 (P0 steps) / -1 (P1 steps).
    var players = [0, 1, 0, 1, 0]
    var expected: List[Float64] = [1.0, -1.0, 1.0, -1.0, 1.0]
    for t in range(5):
        assert_equal(az_value_target(RESULT_P0_WINS, players[t]), expected[t])


def test_zero_sum_sign() raises:
    assert_equal(zero_sum_sign(0, 0), 1.0)
    assert_equal(zero_sum_sign(1, 1), 1.0)
    assert_equal(zero_sum_sign(0, 1), -1.0)
    assert_equal(zero_sum_sign(1, 0), -1.0)


def test_flip_for_perspective() raises:
    # A +0.7 value seen by the opponent is -0.7 from our perspective.
    assert_equal(flip_for_perspective(0.7, 1, 0), -0.7)
    assert_equal(flip_for_perspective(0.7, 0, 0), 0.7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

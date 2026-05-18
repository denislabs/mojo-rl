"""Phase 0 planners: muzero.strategies shim coverage.

Imports every symbol the legacy `mojo_rl.deep_agents.muzero.strategies`
module is expected to re-export, then exercises a handful of static methods
to confirm the comptime params resolve through the shim. If any name is
dropped during the strategies promotion this test fails to compile.

Usage:
    pixi run mojo run -I . tests/planners/test_strategies_shim.mojo
"""

from std.math import abs as math_abs
from std.testing import assert_true, assert_equal

from mojo_rl.deep_agents.muzero.strategies import (
    # ValueEncoding family (lives in planners/common now).
    ValueEncoding, CategoricalEncoding, ScalarEncoding, SymlogEncoding,
    # MCTS-only strategy traits (live in planners/tree_search now).
    SearchMode, LearnedDynamics, TrueGameRules,
    HiddenScaling, MinMaxScale, NoScale,
    ExplorationNoise, DirichletNoise, EpsilonNoise, NoNoise,
    PUCTFormula, MuZeroPUCT, AlphaGoPUCT, UCB1Formula,
    BackupMode, NStepBootstrap, MonteCarloReturn, LambdaReturn,
    PlayerMode, SinglePlayer, SelfPlay,
)


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_search_mode_flags() raises:
    assert_true(LearnedDynamics.USE_LEARNED_DYNAMICS)
    assert_true(not LearnedDynamics.NEEDS_GAME_STATE)
    assert_true(not TrueGameRules.USE_LEARNED_DYNAMICS)
    assert_true(TrueGameRules.NEEDS_GAME_STATE)


def test_puct_formulas_compute_through_shim() raises:
    # MuZero PUCT: log((1 + N + cb) / cb) + ci
    var c = MuZeroPUCT[].compute_c(0.0, 19652.0, 1.25)
    # At parent_visits=0: log(1/cb + 1) + 1.25 ≈ 1.25 + tiny
    assert_true(c >= 1.25)
    # AlphaGoPUCT: always ci.
    assert_true(_approx(AlphaGoPUCT[].compute_c(123.0, 0.0, 2.5), 2.5))
    # UCB1: always c.
    assert_true(_approx(UCB1Formula[].compute_c(999.0, 0.0, 1.414), 1.414))


def test_player_mode_backup_transform() raises:
    # SinglePlayer: reward + gamma * value
    assert_true(
        _approx(SinglePlayer.backup_transform(2.0, 1.0, 0.5), 1.0 + 0.5 * 2.0)
    )
    # SelfPlay: -value (reward/gamma ignored for zero-sum)
    assert_true(_approx(SelfPlay.backup_transform(2.0, 1.0, 0.5), -2.0))


def test_backup_mode_bootstrap_predicate() raises:
    assert_true(NStepBootstrap.should_bootstrap(5, 5, False))
    assert_true(not NStepBootstrap.should_bootstrap(5, 5, True))
    assert_true(not NStepBootstrap.should_bootstrap(3, 5, False))
    assert_true(not MonteCarloReturn.should_bootstrap(5, 5, False))
    assert_true(LambdaReturn[].should_bootstrap(5, 5, False))


def test_noise_param_round_trip() raises:
    # DirichletNoise — defaults.
    assert_equal(DirichletNoise[].NOISE_TYPE, 0)
    assert_true(_approx(DirichletNoise[].NOISE_FRACTION, 0.25))
    assert_true(_approx(DirichletNoise[].NOISE_ALPHA, 0.25))
    # EpsilonNoise — no alpha.
    assert_equal(EpsilonNoise[].NOISE_TYPE, 1)
    # NoNoise — zeros.
    assert_equal(NoNoise.NOISE_TYPE, 2)
    assert_true(_approx(NoNoise.NOISE_FRACTION, 0.0))


def test_value_encoding_flags() raises:
    assert_true(CategoricalEncoding.IS_DISTRIBUTIONAL)
    assert_true(not ScalarEncoding.USE_SCALAR_TRANSFORM)
    assert_true(SymlogEncoding.USE_SCALAR_TRANSFORM)


def test_hidden_scaling_flags() raises:
    assert_true(MinMaxScale.ENABLED)
    assert_true(not NoScale.ENABLED)


def main() raises:
    print("=== Phase 0 planners: muzero.strategies shim coverage ===")
    test_search_mode_flags()
    print("  PASS SearchMode flags")
    test_puct_formulas_compute_through_shim()
    print("  PASS PUCT formulas (compute through shim)")
    test_player_mode_backup_transform()
    print("  PASS PlayerMode backup_transform")
    test_backup_mode_bootstrap_predicate()
    print("  PASS BackupMode bootstrap predicate")
    test_noise_param_round_trip()
    print("  PASS ExplorationNoise param round-trip")
    test_value_encoding_flags()
    print("  PASS ValueEncoding flags through shim")
    test_hidden_scaling_flags()
    print("  PASS HiddenScaling flags")
    print("OK")

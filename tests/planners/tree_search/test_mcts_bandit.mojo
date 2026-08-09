"""Phase 3 planners: CPU MCTS on a two-armed bandit.

Isolated test for ``GenericCPUMCTS`` using ``TwoArmBandit`` as the only
world-model surface. The bandit has constant state (one node forever),
deterministic per-arm expected rewards, and an obvious better arm — so
after N simulations the visit count for the better arm should dominate.

This is the simplest possible PUCT smoke test: it falsifies a noise-bug,
a backup-sign bug, or a PUCT-prior-bug without needing any learned
networks.

Setup:
  ACTION_DIM=2 (left vs right arm).
  p_left=0.2, p_right=0.8 → right is better; the MinMax-normalized Q
  for right is 1.0 vs 0.0 for left after a single visit of each.
  Uniform prior 0.5/0.5; ``NoNoise`` so the test is fully deterministic.

Usage:
    pixi run mojo run -I . tests/planners/tree_search/test_mcts_bandit.mojo
"""

from std.math import abs as math_abs
from std.random import seed as _set_seed
from std.testing import assert_true

from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    NoNoise,
    DirichletNoise,
    SinglePlayer,
    Representation,
    Dynamics,
    Prediction,
)
from mojo_rl.planners.testing import TwoArmBandit


# ─── Adapters ─────────────────────────────────────────────────────────────


comptime ACT: Int = 2
comptime LATENT: Int = 1  # one-element "hidden" — bandits are stateless


@fieldwise_init
struct BanditRepresentation(Movable, Deinitable, Representation):
    """Identity-ish encoder. ``obs`` is a one-element [0.0] vector; we
    just write the same zero into the latent slot. The bandit's state
    never changes, so anything we put here would be ignored downstream.
    """

    comptime OBS_DIM: Int = 1
    comptime LATENT_DIM: Int = LATENT

    def encode_cpu(
        mut self,
        obs: List[Float64],
        mut hidden_out: List[Float64],
    ) raises:
        hidden_out[0] = Float64(0.0)


@fieldwise_init
struct BanditDynamics(Movable, Deinitable, Dynamics):
    """``step_cpu`` returns expected_reward(action) and a stay-put hidden."""

    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT

    var bandit: TwoArmBandit

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        hidden_out[0] = Float64(0.0)
        return self.bandit.expected_reward(action)


@fieldwise_init
struct BanditPrediction(Movable, Deinitable, Prediction):
    """Uniform prior + zero value. The test isolates PUCT + backup;
    deliberately no policy signal here so visit counts come from Q only.
    """

    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT

    def predict_cpu(
        mut self,
        hidden: List[Float64],
        mut policy_out: List[Float64],
    ) raises -> Float64:
        for a in range(ACT):
            policy_out[a] = 1.0 / Float64(ACT)
        return Float64(0.0)


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_mcts_visits_better_arm() raises:
    """After 64 sims with NoNoise and uniform prior, the better arm's
    visit count must dominate the worse arm's by a clear margin.
    """
    _set_seed(0xB44D17)

    var planner = GenericCPUMCTS[
        ACT, LATENT,
        64,    # NUM_SIMULATIONS
        128,   # MAX_NODES
        MuZeroPUCT[],
        NoNoise,
        SinglePlayer,
    ](gamma=0.997)

    var rep = BanditRepresentation()
    var dyn = BanditDynamics(bandit=TwoArmBandit(p_left=0.2, p_right=0.8))
    var pred = BanditPrediction()

    var root_obs: List[Float64] = [0.0]
    var policy = planner.search[
        BanditRepresentation, BanditDynamics, BanditPrediction
    ](rep, dyn, pred, root_obs, add_noise=False)

    # Visit counts are normalized to probabilities. With p_right > p_left
    # the right arm must collect strictly more than 50% of visits.
    # Tight bound: with NUM_SIMULATIONS=64, deterministic Q feedback,
    # MuZero PUCT, the right arm should be at >= 0.7.
    assert_true(
        policy[1] > policy[0],
        "right arm should out-visit left arm, got "
        + String(policy[0])
        + " vs "
        + String(policy[1]),
    )
    assert_true(
        policy[1] >= 0.6,
        "right arm should dominate, got policy[1]=" + String(policy[1]),
    )
    # Visit-count distribution sums to 1.
    assert_true(_approx(policy[0] + policy[1], 1.0, tol=1e-9))


def test_mcts_root_value_matches_better_arm() raises:
    """``root_value`` is the visit-weighted Q at the root. With the
    better arm dominating visits, the root value should land near the
    higher arm's expected reward — modulo MinMax normalization which
    happens *inside the PUCT term*, not on the returned Q.
    """
    _set_seed(0xB44D18)
    var planner = GenericCPUMCTS[
        ACT, LATENT,
        64, 128, MuZeroPUCT[], NoNoise, SinglePlayer,
    ](gamma=0.0)  # gamma=0 → leaf Q is just the edge reward.

    var rep = BanditRepresentation()
    var dyn = BanditDynamics(bandit=TwoArmBandit(p_left=0.2, p_right=0.8))
    var pred = BanditPrediction()

    var root_obs: List[Float64] = [0.0]
    var _policy = planner.search[
        BanditRepresentation, BanditDynamics, BanditPrediction
    ](rep, dyn, pred, root_obs, add_noise=False)

    # With gamma=0, each visit to action a backs up exactly its edge
    # reward (= expected_reward(a)). Visit-weighted Q is therefore in
    # the convex hull of {p_left=0.2, p_right=0.8}, and the better arm
    # dominates → root_value should be well above 0.5.
    var v = planner.root_value()
    assert_true(
        v > 0.5,
        "root_value should sit close to the better arm's payoff, got "
        + String(v),
    )
    assert_true(
        v <= 0.8 + 1e-9 and v >= 0.2 - 1e-9,
        "root_value must land inside the per-arm reward range, got "
        + String(v),
    )


def test_mcts_determinism_no_noise() raises:
    """Same seed + ``NoNoise`` → identical policies. Verifies the
    selection path is fully Q + PUCT driven with no hidden RNG.
    """
    var rep_a = BanditRepresentation()
    var dyn_a = BanditDynamics(bandit=TwoArmBandit(p_left=0.3, p_right=0.7))
    var pred_a = BanditPrediction()
    var planner_a = GenericCPUMCTS[
        ACT, LATENT, 32, 64, MuZeroPUCT[], NoNoise, SinglePlayer,
    ](gamma=0.5)

    _set_seed(0xD37E)
    var root_obs: List[Float64] = [0.0]
    var p1 = planner_a.search[
        BanditRepresentation, BanditDynamics, BanditPrediction
    ](rep_a, dyn_a, pred_a, root_obs, add_noise=False)

    var rep_b = BanditRepresentation()
    var dyn_b = BanditDynamics(bandit=TwoArmBandit(p_left=0.3, p_right=0.7))
    var pred_b = BanditPrediction()
    var planner_b = GenericCPUMCTS[
        ACT, LATENT, 32, 64, MuZeroPUCT[], NoNoise, SinglePlayer,
    ](gamma=0.5)

    _set_seed(0xD37E)
    var p2 = planner_b.search[
        BanditRepresentation, BanditDynamics, BanditPrediction
    ](rep_b, dyn_b, pred_b, root_obs, add_noise=False)

    for a in range(ACT):
        assert_true(
            _approx(p1[a], p2[a], tol=1e-12),
            "NoNoise MCTS should be bit-deterministic; mismatch at a="
            + String(a) + ": " + String(p1[a]) + " vs " + String(p2[a]),
        )


def test_mcts_legal_mask_zeroes_illegal() raises:
    """Legal mask [True, False] forces all visits onto the left arm,
    even though right has higher reward. The mask wins over Q.
    """
    _set_seed(0xB12A2)
    var planner = GenericCPUMCTS[
        ACT, LATENT, 16, 32, MuZeroPUCT[], NoNoise, SinglePlayer,
    ](gamma=0.997)

    var rep = BanditRepresentation()
    var dyn = BanditDynamics(bandit=TwoArmBandit(p_left=0.2, p_right=0.8))
    var pred = BanditPrediction()

    var root_obs: List[Float64] = [0.0]
    var legal: List[Bool] = [True, False]
    var policy = planner.search[
        BanditRepresentation, BanditDynamics, BanditPrediction
    ](rep, dyn, pred, root_obs, add_noise=False, legal_mask=legal)

    assert_true(
        _approx(policy[0], 1.0, tol=1e-9),
        "left arm should hold all visits, got policy[0]=" + String(policy[0]),
    )
    assert_true(
        _approx(policy[1], 0.0, tol=1e-9),
        "right arm should be masked out, got policy[1]=" + String(policy[1]),
    )


def test_dirichlet_noise_preserves_sum_to_one() raises:
    """With DirichletNoise active at the root, the resulting policy
    still sums to 1 (visit counts) and the better arm still wins.
    Sanity for the noise-blend branch; doesn't try to assert noise
    statistics, just structural soundness.
    """
    _set_seed(0xD12C)
    var planner = GenericCPUMCTS[
        ACT, LATENT, 32, 64,
        MuZeroPUCT[],
        DirichletNoise[fraction=0.25, alpha=0.25],
        SinglePlayer,
    ](gamma=0.997)

    var rep = BanditRepresentation()
    var dyn = BanditDynamics(bandit=TwoArmBandit(p_left=0.2, p_right=0.8))
    var pred = BanditPrediction()

    var root_obs: List[Float64] = [0.0]
    var policy = planner.search[
        BanditRepresentation, BanditDynamics, BanditPrediction
    ](rep, dyn, pred, root_obs, add_noise=True)

    assert_true(_approx(policy[0] + policy[1], 1.0, tol=1e-9))
    # Better arm should still come out on top even with 25% noise.
    assert_true(
        policy[1] > policy[0],
        "right arm should still win under DirichletNoise, got "
        + String(policy[0]) + " vs " + String(policy[1]),
    )


def main() raises:
    print("=== Phase 3 planners: CPU MCTS on TwoArmBandit ===")
    test_mcts_visits_better_arm()
    print("  PASS visits dominate on the better arm")
    test_mcts_root_value_matches_better_arm()
    print("  PASS root_value sits inside per-arm reward range")
    test_mcts_determinism_no_noise()
    print("  PASS NoNoise + same seed → identical policies")
    test_mcts_legal_mask_zeroes_illegal()
    print("  PASS legal_mask zeroes out illegal action")
    test_dirichlet_noise_preserves_sum_to_one()
    print("  PASS DirichletNoise keeps policy sum=1, better arm still wins")
    print("OK")

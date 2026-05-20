"""Phase 0 planners: stub world models.

Verifies the stubs themselves are correct so that Phase 1+ planner tests can
rely on them as ground-truth oracles.

Usage:
    pixi run mojo run -I . tests/planners/testing/test_stub_models.mojo
"""

from std.math import abs as math_abs
from std.testing import assert_equal, assert_true

from mojo_rl.planners.testing import (
    IdentityDynamics,
    GoalReachReward,
    LinearQuadratic1D,
    TwoArmBandit,
    KnownValueTree,
)


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_identity_dynamics_pairs_with_goal_reach() raises:
    """If z' = z + a and the reward maximizes at z = goal, then the
    one-step-optimal action from any z is exactly `goal - z` → reward = 0."""
    var z: List[Float64] = [0.0, 0.0, 0.0]
    var goal_vec: List[Float64] = [1.0, -2.0, 3.5]
    var reward_fn = GoalReachReward(goal=goal_vec.copy())

    # Action = goal - z = goal.
    var a: List[Float64] = goal_vec.copy()
    var z_next = IdentityDynamics.step(z, a)

    assert_equal(z_next[0], 1.0)
    assert_equal(z_next[1], -2.0)
    assert_equal(z_next[2], 3.5)

    # At z_next == goal, reward = 0 and distance = 0.
    assert_true(_approx(reward_fn.reward(z_next), 0.0))
    assert_true(_approx(reward_fn.distance(z_next), 0.0))


def test_goal_reach_reward_strictly_concave() raises:
    """Reward at the goal is 0 and strictly dominates any other point."""
    var goal_vec: List[Float64] = [1.0, 1.0]
    var reward_fn = GoalReachReward(goal=goal_vec.copy())
    var r_at_goal = reward_fn.reward(goal_vec.copy())
    assert_true(_approx(r_at_goal, 0.0))
    var off: List[Float64] = [0.5, 1.5]
    var r_off = reward_fn.reward(off)
    assert_true(r_off < r_at_goal)
    # ‖[0.5, 1.5] - [1, 1]‖² = 0.25 + 0.25 = 0.5
    assert_true(_approx(r_off, -0.5))


def test_lqr_gain_for_known_problem() raises:
    """A = 1.0, B = 1.0, Q = 1.0, R = 1.0 is the canonical scalar LQR test.
    The Riccati fixed point gives P = (1 + sqrt(5)) / 2 (golden ratio).
    K = (B P A) / (R + B² P) = P / (1 + P).
    With P = φ ≈ 1.618034 → K ≈ 0.618034.
    """
    var sys = LinearQuadratic1D(A=1.0, B=1.0, Q=1.0, R=1.0)
    var K = sys.lqr_gain_infinite_horizon()
    # Expected K = (sqrt(5) - 1) / 2
    var expected_K = 0.6180339887498949
    assert_true(_approx(K, expected_K, tol=1e-8))


def test_lqr_step_and_reward() raises:
    var sys = LinearQuadratic1D(A=2.0, B=1.0, Q=1.0, R=0.5)
    assert_equal(sys.step(1.0, 0.5), 2.5)
    # r(z=1, a=0.5) = -(1*1 + 0.5*0.25) = -1.125
    assert_true(_approx(sys.reward(1.0, 0.5), -1.125))


def test_two_arm_bandit() raises:
    var b = TwoArmBandit(p_left=0.3, p_right=0.7)
    assert_equal(b.expected_reward(0), 0.3)
    assert_equal(b.expected_reward(1), 0.7)
    assert_equal(b.best_action(), 1)
    # Tie-break: prefer arm 0 when equal (best_action returns 0 if p_left >= p_right).
    var tied = TwoArmBandit(p_left=0.5, p_right=0.5)
    assert_equal(tied.best_action(), 0)


def test_known_value_tree_max() raises:
    # Branching 2, depth 2 → 4 leaves.
    var leaves: List[Float64] = [-1.0, 3.0, 0.0, 2.0]
    var tree = KnownValueTree(branching=2, depth=2, leaf_values=leaves.copy())
    assert_equal(tree.num_leaves(), 4)
    assert_true(_approx(tree.max_value(), 3.0))


def test_known_value_tree_negamax() raises:
    """Branching 2, depth 2, leaves [a b | c d] from root POV after two
    negations:
      level-1 parents: -max(a,b), -max(c,d)
      root           : -max(-max(a,b), -max(c,d)) = -max(-max_left, -max_right)
                     = min(max_left, max_right)
    For leaves [1, 3, 4, 2]: max_left = 3, max_right = 4 → root = min(3, 4) = 3.
    """
    var leaves: List[Float64] = [1.0, 3.0, 4.0, 2.0]
    var tree = KnownValueTree(branching=2, depth=2, leaf_values=leaves.copy())
    assert_true(_approx(tree.negamax_value(), 3.0))


def test_known_value_tree_size_mismatch_raises() raises:
    var leaves: List[Float64] = [1.0, 2.0, 3.0]  # 3 != 2**2
    var tree = KnownValueTree(branching=2, depth=2, leaf_values=leaves.copy())
    var raised = False
    try:
        var _ = tree.negamax_value()
    except:
        raised = True
    assert_true(raised)


def main() raises:
    print("=== Phase 0 planners: stub_models ===")
    test_identity_dynamics_pairs_with_goal_reach()
    print("  PASS IdentityDynamics + GoalReachReward pairing")
    test_goal_reach_reward_strictly_concave()
    print("  PASS GoalReachReward concavity")
    test_lqr_gain_for_known_problem()
    print("  PASS LinearQuadratic1D LQR gain (golden ratio)")
    test_lqr_step_and_reward()
    print("  PASS LinearQuadratic1D step + reward arithmetic")
    test_two_arm_bandit()
    print("  PASS TwoArmBandit")
    test_known_value_tree_max()
    print("  PASS KnownValueTree max")
    test_known_value_tree_negamax()
    print("  PASS KnownValueTree negamax")
    test_known_value_tree_size_mismatch_raises()
    print("  PASS KnownValueTree size-mismatch raises")
    print("OK")

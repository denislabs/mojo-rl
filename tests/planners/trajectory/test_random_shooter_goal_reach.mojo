"""Phase 1 planners: random shooter goal-reach test.

Companion to ``test_cem_goal_reach.mojo``. Same setup
(``IdentityDynamics + GoalReachReward``, BATCH=1, HORIZON=3, ACT_DIM=3,
goal=[1, 1, 1] from origin), only the planner differs.

A goal-reaching plan is any permutation of ``(e_0, e_1, e_2)`` —
3! = 6 plans out of 3³ = 27. With ``num_samples=32`` uniform draws,
the probability of NOT hitting any permutation is ``(21/27)^32 ≈
1.6e-3``, so the test is essentially deterministic.

Usage:
    pixi run mojo run -I . tests/planners/trajectory/test_random_shooter_goal_reach.mojo
"""

from std.math import abs as math_abs
from std.memory import alloc
from std.random import seed as _set_seed
from std.testing import assert_true

from layout import TileTensor, TensorLayout

from mojo_rl.nn.constants import DT as dtype
from mojo_rl.planners.trajectory import (
    CategoricalRandomShooter,
    ScorePlanCallback,
)
from mojo_rl.planners.testing import IdentityDynamics, GoalReachReward


comptime BATCH: Int = 1
comptime HORIZON: Int = 3
comptime ACT_DIM: Int = 3


@fieldwise_init
struct GoalReachScoreCallback(Movable, Deinitable, ScorePlanCallback):
    """Same callback as the CEM test — score plans against a fixed goal
    under z' = z + e_picked dynamics. Duplicated here rather than
    shared so each test file is self-contained.
    """

    var goal_x: Float64
    var goal_y: Float64
    var goal_z: Float64

    def score_plan[L: TensorLayout](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        comptime assert action_plan.flat_rank == 3, (
            "GoalReachScoreCallback expects a 3D (B, H, A) plan"
        )
        var z: List[Float64] = [0.0, 0.0, 0.0]
        for t in range(HORIZON):
            var picked: Int = 0
            for a in range(ACT_DIM):
                if action_plan[0, t, a] > Scalar[dtype](0.5):
                    picked = a
                    break
            var a_vec: List[Float64] = [0.0, 0.0, 0.0]
            a_vec[picked] = 1.0
            z = IdentityDynamics.step(z, a_vec)
        var goal: List[Float64] = [self.goal_x, self.goal_y, self.goal_z]
        var rew = GoalReachReward(goal=goal.copy())
        return -rew.reward(z)


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_shooter_finds_goal_reaching_plan() raises:
    """With 32 uniform draws over 27 possible plans, the shooter is
    essentially guaranteed to hit a goal-reaching permutation. Best
    score should be 0 to within float32 epsilon."""
    _set_seed(0xC0FFEE)
    var shooter = CategoricalRandomShooter[BATCH, ACT_DIM](
        horizon=HORIZON, num_samples=32,
    )
    var cb = GoalReachScoreCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var best_plan = alloc[Scalar[dtype]](BATCH * HORIZON * ACT_DIM).as_unsafe_any_origin()
    var best = shooter.optimize(cb, best_plan, verbose=False)

    assert_true(
        _approx(best, 0.0, tol=1e-6),
        "Shooter should find score=0 plan, got " + String(best),
    )

    # The recovered best plan must decode to a goal-reaching trajectory.
    var z: List[Float64] = [0.0, 0.0, 0.0]
    for t in range(HORIZON):
        var picked: Int = 0
        for a in range(ACT_DIM):
            if best_plan[t * ACT_DIM + a] > Scalar[dtype](0.5):
                picked = a
                break
        var a_vec: List[Float64] = [0.0, 0.0, 0.0]
        a_vec[picked] = 1.0
        z = IdentityDynamics.step(z, a_vec)
    assert_true(_approx(z[0], 1.0))
    assert_true(_approx(z[1], 1.0))
    assert_true(_approx(z[2], 1.0))

    best_plan.free()


def test_sample_scores_populated_for_stats() raises:
    """`sample_scores` must be fully populated after optimize, so callers
    can compute mean / quantile statistics without re-running rollouts.
    """
    _set_seed(0xC0DE)
    var shooter = CategoricalRandomShooter[BATCH, ACT_DIM](
        horizon=HORIZON, num_samples=8,
    )
    var cb = GoalReachScoreCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var best_plan = alloc[Scalar[dtype]](BATCH * HORIZON * ACT_DIM).as_unsafe_any_origin()
    _ = shooter.optimize(cb, best_plan, verbose=False)

    # All slots must be set (scores are ≥ 0 since score = ‖z - goal‖²).
    # Initial fill was 0.0 — a never-overwritten slot would also be 0,
    # which is a valid score. So we instead check that the per-sample
    # values reproduce the manual recompute from `sample_actions`.
    for s in range(8):
        var z: List[Float64] = [0.0, 0.0, 0.0]
        for t in range(HORIZON):
            var picked: Int = 0
            for a in range(ACT_DIM):
                var v = shooter.sample_actions[
                    s * BATCH * HORIZON * ACT_DIM + t * ACT_DIM + a
                ]
                if v > Scalar[dtype](0.5):
                    picked = a
                    break
            var a_vec: List[Float64] = [0.0, 0.0, 0.0]
            a_vec[picked] = 1.0
            z = IdentityDynamics.step(z, a_vec)
        var dx = z[0] - 1.0
        var dy = z[1] - 1.0
        var dz = z[2] - 1.0
        var expected = dx * dx + dy * dy + dz * dz
        assert_true(
            _approx(shooter.sample_scores[s], expected, tol=1e-6),
            "sample_scores["
            + String(s)
            + "] = "
            + String(shooter.sample_scores[s])
            + " != recomputed "
            + String(expected),
        )

    best_plan.free()


def test_ctor_validates_args() raises:
    """Both `horizon < 1` and `num_samples < 1` must raise at construction."""
    var raised_h: Bool = False
    try:
        var _ = CategoricalRandomShooter[BATCH, ACT_DIM](
            horizon=0, num_samples=4,
        )
    except:
        raised_h = True
    assert_true(raised_h, "horizon=0 should raise")

    var raised_n: Bool = False
    try:
        var _ = CategoricalRandomShooter[BATCH, ACT_DIM](
            horizon=2, num_samples=0,
        )
    except:
        raised_n = True
    assert_true(raised_n, "num_samples=0 should raise")


def main() raises:
    print("=== Phase 1 planners: random shooter goal-reach ===")
    test_shooter_finds_goal_reaching_plan()
    print("  PASS shooter finds goal-reaching plan in 32 samples")
    test_sample_scores_populated_for_stats()
    print("  PASS sample_scores populated for downstream stats")
    test_ctor_validates_args()
    print("  PASS ctor validates horizon + num_samples")
    print("OK")

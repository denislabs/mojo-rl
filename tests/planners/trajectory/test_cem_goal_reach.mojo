"""Phase 1 planners: CEM goal-reach test.

Isolated test for `CategoricalCEMOptimizer` using stub world models
(no LeWM agent, no GPU buffers). The trajectory is built by composing
`IdentityDynamics` (z' = z + a) with the unit-vector action e_a
implied by a one-hot CEM sample; the score is `GoalReachReward`'s
distance² to the goal.

Setup:
  BATCH=1, HORIZON=3, ACT_DIM=3, goal=[1, 1, 1] from z0=[0, 0, 0].
  Any permutation of (e_0, e_1, e_2) reaches goal exactly → score = 0.
  CEM should converge in a handful of iterations with cem_topk=8 elites
  out of cem_samples=32 per round.

Usage:
    pixi run mojo run -I . tests/planners/trajectory/test_cem_goal_reach.mojo
"""

from std.math import abs as math_abs
from std.memory import alloc
from std.random import seed as _set_seed
from std.testing import assert_true

from layout import TileTensor, TensorLayout

from mojo_rl.nn.constants import DT as dtype
from mojo_rl.planners.trajectory import (
    CategoricalCEMOptimizer,
    ScorePlanCallback,
)
from mojo_rl.planners.testing import IdentityDynamics, GoalReachReward


# Comptime sizes for this test. The CEM optimizer's comptime params have to
# match the callback's expectations exactly.
comptime BATCH: Int = 1
comptime HORIZON: Int = 3
comptime ACT_DIM: Int = 3


@fieldwise_init
struct GoalReachScoreCallback(Movable, ImplicitlyDeletable, ScorePlanCallback):
    """Score a (1, 3, 3) one-hot plan as MSE-to-goal under z' = z + e_a.

    Each timestep picks one axis (one-hot action e_a); the trajectory
    z_{t+1} = z_t + e_a accumulates over the horizon. Score is
    ``‖z_H - goal‖²``, matching ``GoalReachReward.reward`` up to sign.

    Stateless except for the goal vector — a real agent's callback would
    additionally own its world-model state. Implements
    ``ScorePlanCallback`` so the same CEM optimizer can be re-used.
    """

    var goal_x: Float64
    var goal_y: Float64
    var goal_z: Float64

    def score_plan[L: TensorLayout](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        # BATCH=1: one trajectory. action_plan is (1, HORIZON, ACT_DIM).
        comptime assert action_plan.flat_rank == 3, (
            "GoalReachScoreCallback expects a 3D (B, H, A) plan"
        )
        var z: List[Float64] = [0.0, 0.0, 0.0]
        for t in range(HORIZON):
            # Decode one-hot at timestep t into a unit-vector action.
            var picked: Int = 0
            for a in range(ACT_DIM):
                if action_plan[0, t, a] > Scalar[dtype](0.5):
                    picked = a
                    break
            # Apply IdentityDynamics: z' = z + e_picked.
            var a_vec: List[Float64] = [0.0, 0.0, 0.0]
            a_vec[picked] = 1.0
            z = IdentityDynamics.step(z, a_vec)
        # GoalReachReward.reward returns -‖z - goal‖²; we want the magnitude.
        var goal: List[Float64] = [self.goal_x, self.goal_y, self.goal_z]
        var rew = GoalReachReward(goal=goal.copy())
        return -rew.reward(z)  # = ‖z - goal‖²; lower is better.


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_cem_converges_to_goal() raises:
    """5 CEM iters × 32 samples × topk 8 reach goal exactly (score==0).
    The optimal plan is any permutation of (e_0, e_1, e_2); CEM is
    discovering that the elites' marginals concentrate on a one-hot
    per timestep.
    """
    _set_seed(0xCEB1)
    var planner = CategoricalCEMOptimizer[BATCH, ACT_DIM](
        horizon=HORIZON,
        cem_iters=5,
        cem_samples=32,
        cem_topk=8,
        cem_smoothing=0.25,
    )
    var cb = GoalReachScoreCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var best_plan = alloc[Scalar[dtype]](BATCH * HORIZON * ACT_DIM).as_unsafe_any_origin()
    var best = planner.optimize(cb, best_plan, verbose=False)

    # 32 samples drawn from uniform once is already very likely to include
    # at least one permutation of (e_0, e_1, e_2) — best score should be 0.
    assert_true(
        _approx(best, 0.0, tol=1e-6),
        "CEM best score should hit 0 on goal-reach, got " + String(best),
    )

    # The recovered best_plan must actually decode to a goal-reaching trajectory.
    # best_plan_out is still a raw pointer — the optimizer writes into it via
    # a TileTensor view internally, but the writeback target is a buffer.
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


def test_cem_zero_iters_returns_inf() raises:
    """`cem_iters=0` is the no-op contract: returns +inf, leaves the
    output buffer untouched (we initialize it to a sentinel and check
    the sentinel is still there).
    """
    var planner = CategoricalCEMOptimizer[BATCH, ACT_DIM](
        horizon=HORIZON,
        cem_iters=0,
        cem_samples=4,
        cem_topk=2,
        cem_smoothing=0.5,
    )
    var cb = GoalReachScoreCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var out = alloc[Scalar[dtype]](BATCH * HORIZON * ACT_DIM).as_unsafe_any_origin()
    out[0] = Scalar[dtype](-99.0)  # sentinel
    var best = planner.optimize(cb, out, verbose=False)
    # No iters → no samples scored → best stays at +inf, sentinel preserved.
    assert_true(best > 1e29, "cem_iters=0 should return +inf, got " + String(best))
    assert_true(
        out[0] == Scalar[dtype](-99.0),
        "cem_iters=0 should not overwrite best_plan_out",
    )
    out.free()


def test_cem_topk_bounds_validated() raises:
    """Construction validates `cem_topk` in [1, cem_samples]."""
    var raised: Bool = False
    try:
        var _ = CategoricalCEMOptimizer[BATCH, ACT_DIM](
            horizon=HORIZON,
            cem_iters=1,
            cem_samples=4,
            cem_topk=8,
            cem_smoothing=0.5,
        )
    except:
        raised = True
    assert_true(raised, "cem_topk > cem_samples should raise")


def main() raises:
    print("=== Phase 1 planners: CEM goal-reach ===")
    test_cem_converges_to_goal()
    print("  PASS CEM converges to goal on (IdentityDynamics + GoalReachReward)")
    test_cem_zero_iters_returns_inf()
    print("  PASS cem_iters=0 no-op contract")
    test_cem_topk_bounds_validated()
    print("  PASS cem_topk bounds validation")
    print("OK")

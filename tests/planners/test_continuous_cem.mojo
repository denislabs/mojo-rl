"""ContinuousCEMOptimizer / ContinuousRandomShooter — analytic convergence.

Scores a plan by squared distance to a fixed target plan (aggregated over
the batch). The Gaussian CEM must drive the plan to the target (score → 0)
and beat the random shooter. Pure host-side; no GPU. Validates sample →
elite → refit → converge independent of any world model.

Run:  pixi run mojo run -I . tests/planners/test_continuous_cem.mojo
"""

from layout import TileTensor, TensorLayout, Idx, row_major
from std.testing import assert_true

from mojo_rl.nn2.constants import DT as dtype
from mojo_rl.planners.trajectory.score_callback import ScorePlanCallback
from mojo_rl.planners.trajectory import (
    ContinuousCEMOptimizer,
    ContinuousRandomShooter,
)


comptime BATCH = 2
comptime ACT_DIM = 2
comptime HORIZON = 3


struct QuadScorer(ScorePlanCallback, Movable):
    """score = Σ_{b,t,a} (plan[b,t,a] - target)^2  (lower is better)."""
    var target: List[Scalar[dtype]]   # (BATCH, HORIZON, ACT_DIM)

    def __init__(out self, var target: List[Scalar[dtype]]):
        self.target = target^

    def score_plan[
        L: TensorLayout
    ](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        var tgt = TileTensor(
            self.target,
            row_major((Idx[BATCH], Idx[HORIZON], Idx[ACT_DIM])),
        )
        var s: Float64 = 0.0
        for b in range(BATCH):
            for t in range(HORIZON):
                for a in range(ACT_DIM):
                    var d = Float64(action_plan[b, t, a]) - Float64(tgt[b, t, a])
                    s += d * d
        return s


def main() raises:
    print("=" * 70)
    print("ContinuousCEMOptimizer — analytic convergence")
    print("=" * 70)

    # fixed target plan (in [-1.5, 1.5], reachable from N(0,1))
    var tgt = List[Scalar[dtype]](
        length=BATCH * HORIZON * ACT_DIM, fill=Scalar[dtype](0)
    )
    for i in range(BATCH * HORIZON * ACT_DIM):
        tgt[i] = Scalar[dtype](
            (Float64((i * 2654435761) % 100) / 50.0) - 1.0  # ~[-1, 1]
        )

    var best = List[Scalar[dtype]](
        length=BATCH * HORIZON * ACT_DIM, fill=Scalar[dtype](0)
    )

    # random shooter baseline
    var sc1 = QuadScorer(tgt.copy())
    var rs = ContinuousRandomShooter[BATCH, ACT_DIM](
        horizon=HORIZON, num_samples=300, init_std=1.0
    )
    var rand_min = rs.optimize(sc1, best.unsafe_ptr(), verbose=False)

    # CEM
    var sc2 = QuadScorer(tgt.copy())
    var cem = ContinuousCEMOptimizer[BATCH, ACT_DIM](
        horizon=HORIZON, cem_iters=30, cem_samples=300, cem_topk=30,
        init_std=1.0,
    )
    var cem_best = cem.optimize(sc2, best.unsafe_ptr(), verbose=False)

    print("   random_min=", rand_min, "  cem=", cem_best)
    assert_true(cem_best < rand_min, "CEM beats random shooter")
    assert_true(cem_best < 1e-2, "CEM converges to target (score → 0)")

    # best plan should match the target
    var maxd: Scalar[dtype] = 0.0
    for i in range(BATCH * HORIZON * ACT_DIM):
        var d = (best[i] - tgt[i]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|best_plan - target| =", maxd)
    assert_true(maxd < Scalar[dtype](0.05), "recovered plan ≈ target")

    _ = sc1^; _ = sc2^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

"""Continuous (Gaussian) random shooter — baseline for continuous CEM.

Draws ``num_samples`` plans from a fixed diagonal Gaussian
``N(0, init_std)`` (optionally clamped), scores each via a
``ScorePlanCallback``, returns the minimum and leaves ``sample_scores``
populated for mean / better-than-baseline statistics. The Gaussian analogue
of `CategoricalRandomShooter`; comptime params match `ContinuousCEMOptimizer`
so the two share callers. Used as the planner-works gate baseline in the
LeWM continuous MPC eval (``cem < random_min``).
"""

from layout import TileTensor, Idx, row_major

from mojo_rl.nn.constants import DT as dtype

from .score_callback import ScorePlanCallback
from .continuous_cem import _gauss


struct ContinuousRandomShooter[BATCH: Int, ACT_DIM: Int](
    ImplicitlyDeletable,
    Movable,
):
    var horizon: Int
    var num_samples: Int
    var init_std: Float64
    var clamp_enabled: Bool
    var clamp_lo: Float64
    var clamp_hi: Float64

    var sample_plan: List[Scalar[dtype]]
    var sample_scores: List[Float64]
    """`(num_samples,)` — score of each draw, public for stats."""

    def __init__(
        out self,
        horizon: Int,
        num_samples: Int,
        init_std: Float64 = 1.0,
        clamp_enabled: Bool = False,
        clamp_lo: Float64 = -1.0,
        clamp_hi: Float64 = 1.0,
    ) raises:
        if horizon < 1:
            raise Error("ContinuousRandomShooter: horizon must be >= 1")
        if num_samples < 1:
            raise Error("ContinuousRandomShooter: num_samples must be >= 1")
        self.horizon = horizon
        self.num_samples = num_samples
        self.init_std = init_std
        self.clamp_enabled = clamp_enabled
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        var plan_size = Self.BATCH * horizon * Self.ACT_DIM
        self.sample_plan = List[Scalar[dtype]](
            length=plan_size, fill=Scalar[dtype](0)
        )
        self.sample_scores = List[Float64](length=num_samples, fill=0.0)

    def _clamp(self, v: Float64) -> Float64:
        if not self.clamp_enabled:
            return v
        if v < self.clamp_lo:
            return self.clamp_lo
        if v > self.clamp_hi:
            return self.clamp_hi
        return v

    def optimize[
        CB: ScorePlanCallback
    ](
        mut self,
        mut callback: CB,
        best_plan_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        var best_overall: Float64 = 1.0e30
        var dst = TileTensor(
            best_plan_out,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        for s in range(self.num_samples):
            var plan = TileTensor(
                self.sample_plan,
                row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
            )
            for b in range(Self.BATCH):
                for t in range(self.horizon):
                    for a in range(Self.ACT_DIM):
                        plan[b, t, a] = Scalar[dtype](
                            self._clamp(self.init_std * _gauss())
                        )
            var score = callback.score_plan(plan)
            self.sample_scores[s] = score
            if score < best_overall:
                best_overall = score
                for b in range(Self.BATCH):
                    for t in range(self.horizon):
                        for a in range(Self.ACT_DIM):
                            dst[b, t, a] = self.sample_plan[
                                (b * self.horizon + t) * Self.ACT_DIM + a
                            ]
            if verbose:
                print("    rs sample", s, " score=", score)
        return best_overall

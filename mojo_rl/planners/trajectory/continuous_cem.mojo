"""Continuous (Gaussian) Cross-Entropy Method (CEM) optimizer.

Host-side per-step CEM for CONTINUOUS actions — the Gaussian analogue of
`CategoricalCEMOptimizer`. Maintains a diagonal Gaussian
``mean[BATCH, horizon, ACT_DIM]`` + ``std[...]`` (per batch-row × timestep ×
action dim), samples candidate plans ``a = mean + std·N(0,1)`` (optionally
clamped), scores each via a ``ScorePlanCallback``, then refits mean/std to
the elite (lowest-score) plans. Std is floored at ``min_std`` to keep the
search from collapsing prematurely.

This is the planner the LeWM paper uses for PushT: 300 candidates, 30 CEM
iterations, top-30 elites, initial sampling variance 1 (``init_std=1``),
horizon 5. Like the categorical CEM, the optimizer owns host scratch only;
all GPU work (action upload, latent rollout, score reduction) lives in the
``ScorePlanCallback`` — so `LeWM2MPCScorer` is reused verbatim (it copies
raw float actions, action-representation agnostic).

Batch semantics match `CategoricalCEMOptimizer`: the distribution is
per-batch-row (rows can diverge), the score is one scalar per sample
aggregated across the batch, and elite ranking is over the sample axis.

Usage::

    var planner = ContinuousCEMOptimizer[BATCH, ACT_DIM](
        horizon=5, cem_iters=30, cem_samples=300, cem_topk=30, init_std=1.0,
    )
    var best = planner.optimize(callback, best_plan_out_ptr)
"""

from std.math import log, cos, sqrt
from std.random import random_float64

from layout import TileTensor, Idx, row_major

from mojo_rl.nn2.constants import DT as dtype

from .score_callback import ScorePlanCallback


comptime _TWO_PI: Float64 = 6.283185307179586


def _gauss() -> Float64:
    """One standard-normal sample (Box-Muller; spare partner discarded)."""
    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * log(u1)) * cos(_TWO_PI * u2)


struct ContinuousCEMOptimizer[BATCH: Int, ACT_DIM: Int](
    ImplicitlyDestructible,
    Movable,
):
    """Per-step diagonal-Gaussian CEM optimizer (continuous actions)."""

    var horizon: Int
    var cem_iters: Int
    var cem_samples: Int
    var cem_topk: Int
    var init_std: Float64
    var min_std: Float64
    var clamp_enabled: Bool
    var clamp_lo: Float64
    var clamp_hi: Float64

    var mean: List[Scalar[dtype]]
    """`(BATCH, horizon, ACT_DIM)` — per-step Gaussian mean."""
    var std: List[Scalar[dtype]]
    """`(BATCH, horizon, ACT_DIM)` — per-step Gaussian std."""
    var sample_actions: List[Scalar[dtype]]
    """`(cem_samples, BATCH, horizon, ACT_DIM)` — sampled plans."""
    var sample_plan: List[Scalar[dtype]]
    """`(BATCH, horizon, ACT_DIM)` — current sample (callback layout)."""
    var sample_scores: List[Float64]
    """`(cem_samples,)` — one score per sampled plan."""
    var elite_indices: List[Int]
    """`(cem_topk,)` — indices of top-K samples by lowest score."""

    def __init__(
        out self,
        horizon: Int,
        cem_iters: Int,
        cem_samples: Int,
        cem_topk: Int,
        init_std: Float64 = 1.0,
        min_std: Float64 = 1e-3,
        clamp_enabled: Bool = False,
        clamp_lo: Float64 = -1.0,
        clamp_hi: Float64 = 1.0,
    ) raises:
        if horizon < 1:
            raise Error("ContinuousCEMOptimizer: horizon must be >= 1")
        if cem_iters < 0:
            raise Error("ContinuousCEMOptimizer: cem_iters must be >= 0")
        if cem_samples < 1:
            raise Error("ContinuousCEMOptimizer: cem_samples must be >= 1")
        if cem_topk < 1 or cem_topk > cem_samples:
            raise Error(
                "ContinuousCEMOptimizer: cem_topk must be in [1, cem_samples]"
            )
        if init_std <= 0.0:
            raise Error("ContinuousCEMOptimizer: init_std must be > 0")

        self.horizon = horizon
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_topk = cem_topk
        self.init_std = init_std
        self.min_std = min_std
        self.clamp_enabled = clamp_enabled
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi

        var plan_size = Self.BATCH * horizon * Self.ACT_DIM
        self.mean = List[Scalar[dtype]](length=plan_size, fill=Scalar[dtype](0))
        self.std = List[Scalar[dtype]](length=plan_size, fill=Scalar[dtype](0))
        self.sample_actions = List[Scalar[dtype]](
            length=cem_samples * plan_size, fill=Scalar[dtype](0)
        )
        self.sample_plan = List[Scalar[dtype]](
            length=plan_size, fill=Scalar[dtype](0)
        )
        self.sample_scores = List[Float64](length=cem_samples, fill=0.0)
        self.elite_indices = List[Int](length=cem_topk, fill=-1)

    def _init_dist(mut self):
        var ps = Self.BATCH * self.horizon * Self.ACT_DIM
        for i in range(ps):
            self.mean[i] = Scalar[dtype](0)
            self.std[i] = Scalar[dtype](self.init_std)

    def _clamp(self, v: Float64) -> Float64:
        if not self.clamp_enabled:
            return v
        if v < self.clamp_lo:
            return self.clamp_lo
        if v > self.clamp_hi:
            return self.clamp_hi
        return v

    def _sample_plan(mut self, sample_idx: Int):
        var mean = TileTensor(
            self.mean,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        var std = TileTensor(
            self.std,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (self.cem_samples, Idx[Self.BATCH], self.horizon,
                 Idx[Self.ACT_DIM])
            ),
        )
        var plan = TileTensor(
            self.sample_plan,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                for a in range(Self.ACT_DIM):
                    var z = _gauss()
                    var v = self._clamp(
                        Float64(mean[b, t, a]) + Float64(std[b, t, a]) * z
                    )
                    all_samples[sample_idx, b, t, a] = Scalar[dtype](v)
                    plan[b, t, a] = Scalar[dtype](v)

    def _pick_elites(mut self):
        for k in range(self.cem_topk):
            self.elite_indices[k] = -1
        for k in range(self.cem_topk):
            var best_idx: Int = -1
            var best_score: Float64 = 1.0e30
            for s in range(self.cem_samples):
                var already = False
                for kk in range(k):
                    if self.elite_indices[kk] == s:
                        already = True
                        break
                if not already and self.sample_scores[s] < best_score:
                    best_score = self.sample_scores[s]
                    best_idx = s
            self.elite_indices[k] = best_idx

    def _refit_dist(mut self):
        """Refit per-(b,t,a) Gaussian to the elite samples (MLE mean/std,
        std floored at ``min_std``)."""
        var mean = TileTensor(
            self.mean,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        var std = TileTensor(
            self.std,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (self.cem_samples, Idx[Self.BATCH], self.horizon,
                 Idx[Self.ACT_DIM])
            ),
        )
        var inv_k = 1.0 / Float64(self.cem_topk)
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                for a in range(Self.ACT_DIM):
                    var m: Float64 = 0.0
                    for k in range(self.cem_topk):
                        m += Float64(all_samples[self.elite_indices[k], b, t, a])
                    m *= inv_k
                    var v: Float64 = 0.0
                    for k in range(self.cem_topk):
                        var d = (
                            Float64(all_samples[self.elite_indices[k], b, t, a])
                            - m
                        )
                        v += d * d
                    v *= inv_k
                    var s = sqrt(v)
                    if s < self.min_std:
                        s = self.min_std
                    mean[b, t, a] = Scalar[dtype](m)
                    std[b, t, a] = Scalar[dtype](s)

    def _write_plan(
        mut self, sample: Int,
        best_plan_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (self.cem_samples, Idx[Self.BATCH], self.horizon,
                 Idx[Self.ACT_DIM])
            ),
        )
        var dst = TileTensor(
            best_plan_out,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                for a in range(Self.ACT_DIM):
                    dst[b, t, a] = all_samples[sample, b, t, a]

    def optimize[
        CB: ScorePlanCallback
    ](
        mut self,
        mut callback: CB,
        best_plan_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        """Run `cem_iters` rounds of sample → score → top-K → refit.
        Returns the best score over the whole optimization; writes the
        best-scoring plan into ``best_plan_out`` (BATCH, horizon, ACT_DIM)."""
        self._init_dist()
        var best_overall: Float64 = 1.0e30
        var best_overall_sample: Int = -1

        for cem_it in range(self.cem_iters):
            for s in range(self.cem_samples):
                self._sample_plan(s)
                var plan_view = TileTensor(
                    self.sample_plan,
                    row_major(
                        (Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])
                    ),
                )
                var score = callback.score_plan(plan_view)
                self.sample_scores[s] = score
                if score < best_overall:
                    best_overall = score
                    best_overall_sample = s
            self._pick_elites()
            self._refit_dist()
            if verbose:
                var iter_best: Float64 = 1.0e30
                for s in range(self.cem_samples):
                    if self.sample_scores[s] < iter_best:
                        iter_best = self.sample_scores[s]
                print("    cem iter", cem_it, " best=", iter_best)

        if best_overall_sample >= 0:
            self._write_plan(best_overall_sample, best_plan_out)
        return best_overall

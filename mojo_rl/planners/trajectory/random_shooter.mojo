"""Categorical random shooter — sample N uniform plans, return min score.

The simplest possible ``TrajectoryOptimizer``. Used as a baseline in
MBRL evaluation (PETS, Nagabandi-style MPC) and as a sanity check for
"is my world model action-aware?" diagnostics — if a learned world
model can't beat the random shooter on a held-out task, the model is
not useful for planning.

For each of ``num_samples`` samples, the shooter draws a uniform
one-hot per ``(batch row, timestep)``, scores the resulting plan via
a ``ScorePlanCallback``, and tracks the minimum. After ``optimize``
returns, the caller can also read ``self.sample_scores`` to compute
mean / per-sample statistics (e.g.
``frac_random_worse_than_expert``) — used by LeWM eval to compare
the learned model against a baseline.

Algorithmically this is ``CategoricalCEMOptimizer`` with one iteration
and no elite/refit step; the storage shape and API are intentionally
identical so the two are drop-in interchangeable for any agent that
implements ``ScorePlanCallback``.
"""

from std.random import random_float64

from layout import TileTensor, Idx, row_major

from mojo_rl.nn.constants import DT as dtype

from .score_callback import ScorePlanCallback, BatchedScorePlanCallback


struct CategoricalRandomShooter[BATCH: Int, ACT_DIM: Int](
    Deinitable,
    Movable,
):
    """Uniform-categorical random shooter.

    Draws ``num_samples`` one-hot plans per call, scores each, returns
    the minimum. ``sample_scores`` is left populated after ``optimize``
    so the caller can compute extra statistics without re-running the
    rollouts.

    Comptime params match ``CategoricalCEMOptimizer`` so the two can
    share callers; ``horizon`` and ``num_samples`` are runtime ctor
    args for the same reason CEM's ``horizon`` is runtime (see
    ``cem.mojo`` docstring).
    """

    var horizon: Int
    var num_samples: Int

    var sample_actions: List[Scalar[dtype]]
    """`(num_samples, BATCH, horizon, ACT_DIM)` — sampled one-hot plans."""

    var sample_plan: List[Scalar[dtype]]
    """`(BATCH, horizon, ACT_DIM)` — scratch holding the *current* sample
    plan, layout that ``ScorePlanCallback.score_plan`` expects."""

    var sample_scores: List[Float64]
    """`(num_samples,)` — score of each sampled plan, in draw order.
    Public on purpose: callers (e.g. LeWM eval) read it directly to
    compute mean / better-than-baseline statistics."""

    def __init__(
        out self,
        horizon: Int,
        num_samples: Int,
    ) raises:
        if horizon < 1:
            raise Error("CategoricalRandomShooter: horizon must be >= 1")
        if num_samples < 1:
            raise Error("CategoricalRandomShooter: num_samples must be >= 1")

        self.horizon = horizon
        self.num_samples = num_samples

        var plan_size = Self.BATCH * horizon * Self.ACT_DIM
        self.sample_actions = List[Scalar[dtype]](
            length=num_samples * plan_size, fill=Scalar[dtype](0)
        )
        self.sample_plan = List[Scalar[dtype]](
            length=plan_size, fill=Scalar[dtype](0)
        )
        self.sample_scores = List[Float64](length=num_samples, fill=0.0)

    def _sample_uniform(mut self, sample_idx: Int):
        """Sample one uniform one-hot plan into
        ``sample_actions[sample_idx]`` and ``sample_plan``.
        """
        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (
                    self.num_samples,
                    Idx[Self.BATCH],
                    self.horizon,
                    Idx[Self.ACT_DIM],
                )
            ),
        )
        var plan = TileTensor(
            self.sample_plan,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                var r_act = Int(random_float64() * Float64(Self.ACT_DIM))
                if r_act >= Self.ACT_DIM:
                    r_act = Self.ACT_DIM - 1
                for a in range(Self.ACT_DIM):
                    var v = Scalar[dtype](1.0) if a == r_act else Scalar[dtype](
                        0.0
                    )
                    all_samples[sample_idx, b, t, a] = v
                    plan[b, t, a] = v

    def optimize[
        CB: ScorePlanCallback
    ](
        mut self,
        mut callback: CB,
        best_plan_out: Pointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        """Sample ``num_samples`` random plans, score each, return min.

        Writes the best-scoring plan to ``best_plan_out`` (shape
        ``(BATCH, horizon, ACT_DIM)``). Side-effect: leaves
        ``self.sample_scores`` populated with all per-sample scores so
        the caller can compute mean / quantile / better-than-baseline
        statistics without re-running rollouts.
        """
        var best_overall: Float64 = 1.0e30
        var best_overall_sample: Int = -1

        for s in range(self.num_samples):
            self._sample_uniform(s)
            var plan_view = TileTensor(
                self.sample_plan,
                row_major(
                    (
                        Idx[Self.BATCH],
                        self.horizon,
                        Idx[Self.ACT_DIM],
                    )
                ),
            )
            var score = callback.score_plan(plan_view)
            self.sample_scores[s] = score
            if score < best_overall:
                best_overall = score
                best_overall_sample = s

        if verbose:
            print("    random shooter best=", best_overall)

        if best_overall_sample >= 0:
            var all_samples = TileTensor(
                self.sample_actions,
                row_major(
                    (
                        self.num_samples,
                        Idx[Self.BATCH],
                        self.horizon,
                        Idx[Self.ACT_DIM],
                    )
                ),
            )
            var dst = TileTensor(
                best_plan_out,
                row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
            )
            for b in range(Self.BATCH):
                for t in range(self.horizon):
                    for a in range(Self.ACT_DIM):
                        dst[b, t, a] = all_samples[best_overall_sample, b, t, a]

        return best_overall

    def optimize_batched[
        CB: BatchedScorePlanCallback
    ](
        mut self,
        mut callback: CB,
        best_plan_out: Pointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        """Sample ``num_samples`` random plans, score them in a single
        batched GPU call, return min.

        Same semantics as ``optimize`` (writes best plan + leaves
        ``sample_scores`` populated for caller stats) but the score loop
        is one batched call into the world model instead of
        ``num_samples`` sequential calls. Used by LeWM eval at paper
        config where the per-sample host sync was the bottleneck.
        """
        for s in range(self.num_samples):
            self._sample_uniform(s)

        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (
                    self.num_samples,
                    Idx[Self.BATCH],
                    self.horizon,
                    Idx[Self.ACT_DIM],
                )
            ),
        )
        callback.score_plans_batched(all_samples, self.sample_scores)

        var best_overall: Float64 = 1.0e30
        var best_overall_sample: Int = -1
        for s in range(self.num_samples):
            var score = self.sample_scores[s]
            if score < best_overall:
                best_overall = score
                best_overall_sample = s

        if verbose:
            print("    random shooter best=", best_overall)

        if best_overall_sample >= 0:
            var dst = TileTensor(
                best_plan_out,
                row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
            )
            for b in range(Self.BATCH):
                for t in range(self.horizon):
                    for a in range(Self.ACT_DIM):
                        dst[b, t, a] = all_samples[best_overall_sample, b, t, a]

        return best_overall

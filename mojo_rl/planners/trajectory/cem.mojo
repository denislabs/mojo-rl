"""Categorical Cross-Entropy Method (CEM) optimizer.

Host-side per-step categorical CEM for discrete actions. Maintains
``action_dist[BATCH, horizon, ACT_DIM]`` (per batch-row × per timestep ×
per action probability mass), samples candidate plans, scores each via
a ``ScorePlanCallback``, then refits the distribution to the top-K
plans using add-K (Laplace) smoothing.

Promoted from ``mojo_rl/experimental/lewm/kernels._run_cem_eval_iter``.
The original was specialized to LeWM's autoregressive MPC rollout; this
version is agent-agnostic — the scoring step delegates to a callback,
so LeWM, future MBPO-CEM, or any other discrete-action world-model
agent can reuse the same struct.

**Batch semantics**: action distribution is per-batch-row (different
rows can converge to different plans), but the **score is one scalar
per sample, aggregated across the batch**. Elite selection ranks
across the sample axis only — this matches the LeWM convention where
the world-model rollout produces one MSE-to-goal per (sample, batch
row) and the optimizer pools them.

**Why HORIZON is runtime, not comptime.** Agents like LeWM expose the
plan length (``mpc_horizon`` / ``needed_actions``) as a runtime ctor
arg; pushing it to comptime would cascade into every container struct
(``CEMPlanner``, ``LeWMEvalSuite``, ``LeWMTrainer``). The host-side
CEM loop is not perf-critical compared to the GPU rollout it scores,
so the lost constant-folding is negligible.

Usage::

    var planner = CategoricalCEMOptimizer[BATCH, ACT_DIM](
        horizon=4, cem_iters=5, cem_samples=64, cem_topk=8, cem_smoothing=0.5,
    )
    var callback = MyAgentScoreCallback(...)
    var best = planner.optimize(callback, best_plan_out_ptr)

See ``docs/PLANNERS_PACKAGE.md`` Phase 1.
"""

from std.memory import alloc
from std.random import random_float64

from mojo_rl.nn.constants import dtype

from .score_callback import ScorePlanCallback


struct CategoricalCEMOptimizer[BATCH: Int, ACT_DIM: Int](
    Movable, ImplicitlyDestructible,
):
    """Per-step categorical CEM optimizer.

    Owns host scratch only — no GPU buffers. GPU work (action upload,
    model rollout, score reduction) lives inside the
    ``ScorePlanCallback`` the caller supplies.

    Comptime params fix the batch + action dims; ``horizon`` and CEM
    hyperparameters come in as runtime ctor args.
    """

    var horizon: Int
    var cem_iters: Int
    var cem_samples: Int
    var cem_topk: Int
    var cem_smoothing: Float64

    # Host scratch (raw allocations; freed in __del__).
    var action_dist: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    """`(BATCH, horizon, ACT_DIM)` — per-batch-row categorical at each step."""

    var sample_actions: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    """`(cem_samples, BATCH, horizon, ACT_DIM)` — sampled one-hot plans."""

    var sample_plan: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    """`(BATCH, horizon, ACT_DIM)` — scratch holding the *current* sample
    plan, layout that ``ScorePlanCallback.score_plan`` expects."""

    var sample_scores: UnsafePointer[Float64, MutAnyOrigin]
    """`(cem_samples,)` — one score per sampled plan."""

    var elite_indices: UnsafePointer[Int, MutAnyOrigin]
    """`(cem_topk,)` — indices of top-K samples by lowest score."""

    def __init__(
        out self,
        horizon: Int,
        cem_iters: Int,
        cem_samples: Int,
        cem_topk: Int,
        cem_smoothing: Float64 = 0.5,
    ) raises:
        if horizon < 1:
            raise Error("CategoricalCEMOptimizer: horizon must be >= 1")
        if cem_iters < 0:
            raise Error("CategoricalCEMOptimizer: cem_iters must be >= 0")
        if cem_samples < 1:
            raise Error("CategoricalCEMOptimizer: cem_samples must be >= 1")
        if cem_topk < 1 or cem_topk > cem_samples:
            raise Error(
                "CategoricalCEMOptimizer: cem_topk must be in [1, cem_samples]"
            )
        if cem_smoothing < 0.0:
            raise Error(
                "CategoricalCEMOptimizer: cem_smoothing must be >= 0"
            )

        self.horizon = horizon
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_topk = cem_topk
        self.cem_smoothing = cem_smoothing

        var plan_size = Self.BATCH * horizon * Self.ACT_DIM
        self.action_dist = alloc[Scalar[dtype]](plan_size)
        self.sample_actions = alloc[Scalar[dtype]](cem_samples * plan_size)
        self.sample_plan = alloc[Scalar[dtype]](plan_size)
        self.sample_scores = alloc[Float64](cem_samples)
        self.elite_indices = alloc[Int](cem_topk)

    def __init__(out self, *, deinit take: Self):
        self.horizon = take.horizon
        self.cem_iters = take.cem_iters
        self.cem_samples = take.cem_samples
        self.cem_topk = take.cem_topk
        self.cem_smoothing = take.cem_smoothing
        self.action_dist = take.action_dist
        self.sample_actions = take.sample_actions
        self.sample_plan = take.sample_plan
        self.sample_scores = take.sample_scores
        self.elite_indices = take.elite_indices
        # `deinit` skips take.__del__, so the buffers won't be double-freed.

    def __del__(deinit self):
        if Int(self.action_dist) != 0:
            self.action_dist.free()
        if Int(self.sample_actions) != 0:
            self.sample_actions.free()
        if Int(self.sample_plan) != 0:
            self.sample_plan.free()
        if Int(self.sample_scores) != 0:
            self.sample_scores.free()
        if Int(self.elite_indices) != 0:
            self.elite_indices.free()

    def _init_uniform_dist(mut self):
        var inv_act = Scalar[dtype](1.0 / Float64(Self.ACT_DIM))
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                for a in range(Self.ACT_DIM):
                    self.action_dist[
                        b * self.horizon * Self.ACT_DIM + t * Self.ACT_DIM + a
                    ] = inv_act

    def _sample_plan(mut self, sample_idx: Int):
        """Sample one-hot plan from `action_dist`. Writes:
          - `sample_actions[sample_idx, :, :, :]` for later elite recall.
          - `sample_plan[:, :, :]` so the caller can hand it to the
            score callback verbatim.
        """
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                var r = random_float64()
                var cumul: Float64 = 0.0
                var picked = Self.ACT_DIM - 1
                for a in range(Self.ACT_DIM):
                    cumul += Float64(
                        self.action_dist[
                            b * self.horizon * Self.ACT_DIM
                            + t * Self.ACT_DIM + a
                        ]
                    )
                    if r < cumul:
                        picked = a
                        break
                for a in range(Self.ACT_DIM):
                    var v = (
                        Scalar[dtype](1.0)
                        if a == picked
                        else Scalar[dtype](0.0)
                    )
                    self.sample_actions[
                        (sample_idx * Self.BATCH + b)
                        * self.horizon * Self.ACT_DIM
                        + t * Self.ACT_DIM + a
                    ] = v
                    self.sample_plan[
                        b * self.horizon * Self.ACT_DIM
                        + t * Self.ACT_DIM + a
                    ] = v

    def _pick_elites(mut self):
        # Mark all elite slots as -1, then greedy-pick the lowest score
        # K times (skipping already-picked indices). O(K * S) — fine for
        # typical K ≤ 16 and S ≤ 256.
        for k in range(self.cem_topk):
            self.elite_indices[k] = -1
        for k in range(self.cem_topk):
            var best_idx: Int = -1
            var best_score: Float64 = 1.0e30
            for s in range(self.cem_samples):
                var already_picked = False
                for kk in range(k):
                    if self.elite_indices[kk] == s:
                        already_picked = True
                        break
                if (
                    not already_picked
                    and self.sample_scores[s] < best_score
                ):
                    best_score = self.sample_scores[s]
                    best_idx = s
            self.elite_indices[k] = best_idx

    def _refit_dist(mut self):
        """Refit per-step categorical from elites with add-K smoothing.

        action_dist[b, t, a] = (count[b, t, a] + smoothing)
                                / (topk + ACT_DIM * smoothing)
        Effectively a Dirichlet-conjugate posterior with uniform prior.
        """
        var denom = (
            Float64(self.cem_topk)
            + Float64(Self.ACT_DIM) * self.cem_smoothing
        )
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                # Reset to prior (smoothing / denom).
                for a in range(Self.ACT_DIM):
                    self.action_dist[
                        b * self.horizon * Self.ACT_DIM + t * Self.ACT_DIM + a
                    ] = Scalar[dtype](self.cem_smoothing / denom)
                # Add 1/denom for each elite vote at this (b, t).
                for k in range(self.cem_topk):
                    var e = self.elite_indices[k]
                    for a in range(Self.ACT_DIM):
                        var v = self.sample_actions[
                            (e * Self.BATCH + b)
                            * self.horizon * Self.ACT_DIM
                            + t * Self.ACT_DIM + a
                        ]
                        if v > Scalar[dtype](0.5):
                            self.action_dist[
                                b * self.horizon * Self.ACT_DIM
                                + t * Self.ACT_DIM + a
                            ] += Scalar[dtype](1.0 / denom)
                            break

    def optimize[CB: ScorePlanCallback](
        mut self,
        mut callback: CB,
        best_plan_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        """Run `cem_iters` rounds of sample → score → top-K → refit.

        Returns the best score observed over the entire optimization.
        Writes the best-scoring plan into ``best_plan_out`` (shape
        ``(BATCH, horizon, ACT_DIM)``).

        ``best_plan_out`` may be the same buffer the caller already owns
        for the next pipeline stage — the writeback is a single
        ``BATCH * horizon * ACT_DIM`` copy.
        """
        self._init_uniform_dist()

        var best_overall: Float64 = 1.0e30
        var best_overall_sample: Int = -1

        for cem_it in range(self.cem_iters):
            # Sample + score.
            for s in range(self.cem_samples):
                self._sample_plan(s)
                var score = callback.score_plan(self.sample_plan)
                self.sample_scores[s] = score
                if score < best_overall:
                    best_overall = score
                    best_overall_sample = s

            # Elites + refit.
            self._pick_elites()
            self._refit_dist()

            if verbose:
                var iter_best: Float64 = 1.0e30
                for s in range(self.cem_samples):
                    if self.sample_scores[s] < iter_best:
                        iter_best = self.sample_scores[s]
                print("    cem iter", cem_it, " best=", iter_best)

        # Write the best plan over the entire optimization back to caller.
        # If cem_iters == 0 (no optimization), leave best_plan_out untouched
        # and return +inf to signal "nothing optimized".
        if best_overall_sample >= 0:
            var plan_size = Self.BATCH * self.horizon * Self.ACT_DIM
            var base = best_overall_sample * plan_size
            for i in range(plan_size):
                best_plan_out[i] = self.sample_actions[base + i]

        return best_overall

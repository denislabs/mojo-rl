"""ScorePlanCallback trait — host-side rollout score contract.

Used by host-driven trajectory optimizers (currently
``CategoricalCEMOptimizer``) that need to evaluate candidate action
plans without owning the world model. The optimizer hands the callback
a ``(BATCH, HORIZON, ACT_DIM)`` host-side plan and gets back one
Float64 — typically a sum/mean of per-batch-row rewards or losses.
Everything GPU-side (action upload, model rollout, score reduction)
lives inside the callback.

The dimensions ``BATCH``, ``HORIZON``, ``ACT_DIM`` are implicit in the
contract — both the optimizer and the callback are constructed against
the same agent config, so the buffer layout is fixed by construction.

Concrete implementations live next to the agent that owns the world
model — e.g. ``mojo_rl/experimental/lewm/lewm_rollout_callback.mojo``
wraps LeWM's autoregressive MPC shot. New planners (MPPI, iLQR) can
reuse the same trait when they need host-side scoring; planners that
keep the rollout fully on-device (the planned ``GPUMCTS``) don't use
this trait.

The intent is intentionally narrow: this is **not** the full
``RolloutCallback`` sketched in ``docs/PLANNERS_PACKAGE.md`` (which would
expose a per-step ``rollout_step_gpu`` contract for direct on-device
optimization). That richer contract is deferred to Phase 2/4. For
host-driven CEM, scoring an entire plan in one call is enough.
"""

from mojo_rl.nn.constants import dtype


trait ScorePlanCallback(ImplicitlyDestructible):
    """Score a candidate action plan for trajectory optimization.

    The callback owns whatever scratch is needed to evaluate a plan
    (GPU buffers, world-model views, etc.). The optimizer treats it as
    a black box: pass a host plan, receive a scalar.

    Implementations must agree with the caller on the
    ``(BATCH, HORIZON, ACT_DIM)`` shape of ``action_plan_host``. A
    lower score is better by convention — e.g. CEM picks the K plans
    with the smallest score values.
    """

    def score_plan(
        mut self,
        action_plan_host: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises -> Float64:
        """Score `action_plan_host` of shape (BATCH, HORIZON, ACT_DIM).

        Returns a single scalar — typically sum-of-MSEs or sum-of-rewards
        aggregated across the batch. Lower is better.
        """
        ...

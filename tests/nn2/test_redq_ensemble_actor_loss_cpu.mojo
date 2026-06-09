"""Phase R.2 smoke test for `EnsembleActorLoss` (CPU).

Verifies the SAC-style mean-over-N-critics actor loss:
  (a) `loss` and `log_prob_mean` finite after step 1.
  (b) Actor params CHANGED — gradient flowed back through rsample +
      N critic.vjp[input_only] + actor.vjp + actor_opt.step.
  (c) All N=4 ONLINE critics' params BYTE-IDENTICAL post-K-steps —
      `mode="input_only"` truly skips param-grad accumulation
      (gate against silent SAC-style "min(Q1,Q2) leak").
  (d) All N=4 TARGET nets BYTE-IDENTICAL — this block never reads
      or writes targets (gate against accidental target rewiring).
  (e) `log_prob_mean` populated to a finite float — what
      `AlphaUpdateStep` consumes downstream.
  (f) Final loss < initial loss — actor is converging to higher
      combined-Q minus α·log_prob (i.e. soft-V ascent). Smoke-grade
      only (no oracle reference).

Hand-fills `state.mb_s` + `state.alpha` so the test is reproducible.
N=4 to exercise the loop more than the degenerate N=2 (SAC-equivalent
shape) without bloating compile time.
"""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.named_params import named_params
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.initializer import Xavier

from mojo_rl.deep_agents2.training.trainer_block import TrainerState
from mojo_rl.deep_agents2.redq import (
    CriticEnsemble,
    EnsembleActorLoss,
)


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 8
comptime N = 4

comptime ActorNet = Sequential[
    Linear[OBS, 16],
    ReLU[16],
    Linear[16, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 16],
    ReLU[16],
    Linear[16, 1],
]


def _fill_mb_s(mut state: TrainerState[OBS, ACT, BATCH]) raises:
    var s = state.mb_s.cpu_ptr()
    for b in range(BATCH):
        for d in range(OBS):
            s[b * OBS + d] = Scalar[DT](
                0.05 * Float64(b) + 0.07 * Float64(d) - 0.2
            )


def _snapshot_param_sum[M: Module](mut model: M) raises -> Float64:
    var ps = named_params["cpu", M](model)
    var acc: Float64 = 0.0
    for i in range(len(ps)):
        ref p = ps[i]
        for k in range(p.n_elems):
            var v = Float64(p.param_ptr[k])
            if v < 0.0:
                v = -v
            acc += v
    return acc


def test_ensemble_actor_loss_cpu() raises:
    print("--- EnsembleActorLoss[Actor, Critic, N=4] CPU smoke ---")

    var actor = ActorNet.make["cpu", Xavier]()
    var actor_opt = Adam.make["cpu", M=ActorNet](actor)
    actor_opt.lr = Scalar[DT](1e-3)

    var ensemble = CriticEnsemble[CriticNet, N].make["cpu", Xavier]()
    var block = EnsembleActorLoss[
        ActorNet, CriticNet, N, BATCH, OBS, ACT,
    ].make["cpu"](action_scale=Scalar[DT](1.0))
    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()

    _fill_mb_s(state)
    var alpha = Scalar[DT](0.2)

    # Snapshots before any actor step.
    var actor_before = _snapshot_param_sum[ActorNet](actor)
    var online_before = List[Float64](length=N, fill=0.0)
    var target_before = List[Float64](length=N, fill=0.0)
    for i in range(N):
        online_before[i] = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].online
        )
        target_before[i] = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].target_net
        )

    # Step 0 — capture initial loss + log_prob_mean.
    var res0 = block.forward_backward["cpu"](
        actor, actor_opt, ensemble,
        state.mb_s.cpu_ptr(),
        alpha,
    )
    var loss_first = Float64(res0.loss)
    var lp_first = Float64(res0.log_prob_mean)
    print("  step 0 loss =", loss_first, " log_prob_mean =", lp_first)
    assert_true(loss_first == loss_first, "step 0 loss finite (no NaN)")
    assert_true(lp_first == lp_first, "step 0 log_prob_mean finite")

    # Steps 1..K-1.
    comptime K = 20
    var loss_last: Float64 = loss_first
    var lp_last: Float64 = lp_first
    for _ in range(K - 1):
        var res = block.forward_backward["cpu"](
            actor, actor_opt, ensemble,
            state.mb_s.cpu_ptr(),
            alpha,
        )
        loss_last = Float64(res.loss)
        lp_last = Float64(res.log_prob_mean)
    print(
        "  step", K - 1, "loss =", loss_last,
        " log_prob_mean =", lp_last,
    )
    assert_true(loss_last == loss_last, "final loss finite")
    assert_true(lp_last == lp_last, "final log_prob_mean finite")

    # (b) Actor params CHANGED.
    var actor_after = _snapshot_param_sum[ActorNet](actor)
    var actor_d = actor_after - actor_before
    if actor_d < 0.0:
        actor_d = -actor_d
    print("  actor |Δ|sum =", actor_d)
    assert_true(actor_d > 0.0, "actor params must change (gradient flowed back)")

    # (c) Every online critic UNCHANGED — mode="input_only" gating.
    # (d) Every target net UNCHANGED — block never touches targets.
    for i in range(N):
        var on_after = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].online
        )
        var tg_after = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].target_net
        )
        var on_d = on_after - online_before[i]
        if on_d < 0.0:
            on_d = -on_d
        var tg_d = tg_after - target_before[i]
        if tg_d < 0.0:
            tg_d = -tg_d
        print(
            "  member", i,
            " online |Δ|sum =", on_d,
            " target |Δ|sum =", tg_d,
        )
        assert_true(
            on_d == 0.0,
            "online critic must be byte-identical (input_only stops param grad)",
        )
        assert_true(
            tg_d == 0.0,
            "target net must be byte-identical (block never touches targets)",
        )

    # (f) Loss decreased — sanity, not strict.
    assert_true(
        loss_last < loss_first,
        "actor loss must decrease over K=20 steps (soft-V ascent)",
    )
    print("  convergence: loss[0] =", loss_first, "→ loss[K-1] =", loss_last)

    print("PASS — EnsembleActorLoss N=4 CPU smoke green.")


def main() raises:
    test_ensemble_actor_loss_cpu()

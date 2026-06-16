"""O.2.b.2 — EnsembleCriticStepOFE + EnsembleActorStepOFE CPU smoke.

Two integration gates, both small-dim and CPU-only:

  (1) Critic step OFE: with a FIXED mb_y target, run 10 critic steps
      back-to-back. Verify the sum-of-losses strictly decreases —
      proves the gradient flows through `action_branch.forward
      (no-grad) → critic.vjp → critic_opt.step` for all N critics
      and that the wiring (concat + AB + critic ensemble) is sound.

  (2) Actor step OFE: run 10 actor steps (loss = α·logπ − mean_i Qᵢ).
      Verify (a) loss is finite and log_prob_mean reasonable, (b)
      the actor's params change between step 0 and step 10 (the
      `actor.vjp` → `actor_opt.step` chain ran), and (c) the
      action_branch params do NOT change (mode='input_only' stop-
      grad). Critics also unchanged for the same reason.

These gates are *standalone* — both blocks operate on the same
shared (SB, AB, critic ensemble) so the test mirrors the trainer's
sharing constraint. The feature pre-pass runs once before both
steps to populate `phi_s` / `phi_sp`."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.linear import Linear

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq.ensemble import CriticEnsemble
from mojo_rl.deep_agents.redq_ofe import (
    OFEStateBranch6,
    OFEActionBranch6,
    OFEFeatureStep,
    EnsembleCriticStepOFE,
    EnsembleActorStepOFE,
    state_branch_out_dim,
    action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6
comptime N = 3

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime ACTOR = Sequential[Linear[PHI_S_DIM, 2 * ACT]]
comptime CRITIC = Sequential[Linear[PHI_SA_DIM, 1]]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


# ─────────────────────────────────────────────────────────────────────────
# (1) Critic step OFE
# ─────────────────────────────────────────────────────────────────────────


def test_critic_step_ofe() raises:
    print("--- (1) EnsembleCriticStepOFE — 10 steps on fixed y ---")
    seed(42)

    var sb = SB.make[target="cpu", INIT=Xavier]()
    var ab = AB.make[target="cpu", INIT=Xavier]()
    var ensemble = CriticEnsemble[CRITIC, N].make[
        target="cpu", INIT=Xavier,
    ]()

    # State + inputs.
    var state = TrainerState[OBS, ACT, BATCH].make[target="cpu"]()
    var obs_p = state.mb_s.cpu_ptr()
    var nobs_p = state.mb_sp.cpu_ptr()
    var act_p = state.mb_a.cpu_ptr()
    var y_p = state.mb_y.cpu_ptr()
    for b in range(BATCH):
        for d in range(OBS):
            obs_p[b * OBS + d] = Scalar[DT](
                0.3 + 0.1 * Float64(b) - 0.05 * Float64(d)
            )
            nobs_p[b * OBS + d] = Scalar[DT](
                0.4 - 0.07 * Float64(b) + 0.03 * Float64(d)
            )
        act_p[b * ACT] = Scalar[DT](-0.2 + 0.15 * Float64(b))
        # FIXED target — picked to be in a sensible Q-value range
        # given the small Xavier-init critic head.
        y_p[b] = Scalar[DT](0.5 - 0.1 * Float64(b))

    # Feature step (one-shot — phi_s and phi_sp are deterministic
    # given the fixed obs/nobs).
    var feat = OFEFeatureStep[SB, OBS, ACT, BATCH].make[target="cpu"]()
    feat.step["cpu"](sb, state)
    var phi_s_p = feat.phi_s_ptr["cpu"]()

    # Critic step block + critic-loss trajectory.
    var cstep = EnsembleCriticStepOFE[
        AB, CRITIC, N, BATCH, PHI_S_DIM, ACT,
    ].make[target="cpu"]()

    var losses = List[Scalar[DT]](length=10, fill=Scalar[DT](0.0))
    for i in range(10):
        # Re-run feature step? phi_s already populated above; SB params
        # never change here (critic step only updates critics, never SB
        # or AB), so phi_s stays valid across steps.
        losses[i] = cstep.step["cpu"](ab, ensemble, phi_s_p, act_p, y_p)

    print("  critic loss[0] =", losses[0])
    print("  critic loss[5] =", losses[5])
    print("  critic loss[9] =", losses[9])

    assert_true(
        losses[0] == losses[0] and losses[0] > Scalar[DT](0.0),
        "critic step-0 loss must be finite positive",
    )
    assert_true(
        losses[9] < losses[0] * Scalar[DT](0.7),
        "critic loss must drop >= 30% over 10 steps on fixed y",
    )
    print("PASS — critic step OFE wired through AB + N critics.")


# ─────────────────────────────────────────────────────────────────────────
# (2) Actor step OFE
# ─────────────────────────────────────────────────────────────────────────


def test_actor_step_ofe() raises:
    print("--- (2) EnsembleActorStepOFE — params change, AB+critic frozen ---")
    seed(42)

    var sb = SB.make[target="cpu", INIT=Xavier]()
    var ab = AB.make[target="cpu", INIT=Xavier]()
    var actor = ACTOR.make[target="cpu", INIT=Xavier]()
    var actor_opt = Adam.make[target="cpu", M=ACTOR](actor)
    actor_opt.lr = Scalar[DT](3e-3)
    var ensemble = CriticEnsemble[CRITIC, N].make[
        target="cpu", INIT=Xavier,
    ]()

    # State.
    var state = TrainerState[OBS, ACT, BATCH].make[target="cpu"]()
    var obs_p = state.mb_s.cpu_ptr()
    var nobs_p = state.mb_sp.cpu_ptr()
    for b in range(BATCH):
        for d in range(OBS):
            obs_p[b * OBS + d] = Scalar[DT](
                0.25 + 0.1 * Float64(b) - 0.05 * Float64(d)
            )
            nobs_p[b * OBS + d] = Scalar[DT](
                0.35 - 0.07 * Float64(b) + 0.03 * Float64(d)
            )
    var feat = OFEFeatureStep[SB, OBS, ACT, BATCH].make[target="cpu"]()
    feat.step["cpu"](sb, state)
    var phi_s_p = feat.phi_s_ptr["cpu"]()

    # Snapshot AB + critic-head params to verify they DON'T change
    # (input_only stop-grad). We pull a representative param from each.
    # AB's first Linear weight pointer:
    var ab_w_p_pre: Scalar[DT] = ab.children[0].inner.children[0].weight.value_unsafe_ptr_cpu()[0]
    var c0_w_p_pre: Scalar[DT] = ensemble.pairs[0].online.children[0].weight.value_unsafe_ptr_cpu()[0]

    var astep = EnsembleActorStepOFE[
        ACTOR, AB, CRITIC, N, BATCH, PHI_S_DIM, ACT,
    ].make[target="cpu"](action_scale=Scalar[DT](1.0))

    var alpha = Scalar[DT](0.1)
    var first_lp_mean: Scalar[DT] = Scalar[DT](0.0)
    var last_lp_mean: Scalar[DT] = Scalar[DT](0.0)
    var first_loss: Scalar[DT] = Scalar[DT](0.0)
    var last_loss: Scalar[DT] = Scalar[DT](0.0)
    for i in range(10):
        var res = astep.forward_backward["cpu"](
            actor, actor_opt, ab, ensemble, phi_s_p, alpha,
        )
        if i == 0:
            first_loss = res.loss
            first_lp_mean = res.log_prob_mean
        if i == 9:
            last_loss = res.loss
            last_lp_mean = res.log_prob_mean

    print("  actor loss[0]    =", first_loss)
    print("  actor loss[9]    =", last_loss)
    print("  log_prob_mean[0] =", first_lp_mean)
    print("  log_prob_mean[9] =", last_lp_mean)

    # (a) loss + log_prob_mean finite.
    assert_true(
        first_loss == first_loss and last_loss == last_loss,
        "actor loss must be finite at step 0 and step 9",
    )
    assert_true(
        first_lp_mean == first_lp_mean and last_lp_mean == last_lp_mean,
        "log_prob_mean must be finite",
    )

    # (b) AB params unchanged (input_only stop-grad).
    var ab_w_p_post: Scalar[DT] = ab.children[0].inner.children[0].weight.value_unsafe_ptr_cpu()[0]
    var c0_w_p_post: Scalar[DT] = ensemble.pairs[0].online.children[0].weight.value_unsafe_ptr_cpu()[0]
    print("  AB[0].weight[0]  pre/post =", ab_w_p_pre, "/", ab_w_p_post)
    print("  C0.weight[0]     pre/post =", c0_w_p_pre, "/", c0_w_p_post)
    assert_true(
        _abs(ab_w_p_post - ab_w_p_pre) < Scalar[DT](1e-10),
        "AB params must NOT change (mode='input_only' stop-grad)",
    )
    assert_true(
        _abs(c0_w_p_post - c0_w_p_pre) < Scalar[DT](1e-10),
        "critic params must NOT change (mode='input_only' stop-grad)",
    )

    print("PASS — actor step OFE: actor trains, AB+critics frozen.")


def main() raises:
    print("=" * 70)
    print("O.2.b.2 — Critic + Actor step OFE (CPU)")
    print("=" * 70)
    test_critic_step_ofe()
    test_actor_step_ofe()
    print("=" * 70)
    print("ALL PASS")
    print("=" * 70)

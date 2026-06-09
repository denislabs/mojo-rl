"""O.3+ — REDQOFEMetrics + flush_metrics surface (CPU).

Gates the accumulator behavior on REDQOFETrainer / REDQOFEAgent:

  (1) Fresh agent → `flush_metrics` returns zeros (counts=0, all
      mean fields=0.0 sentinel; no NaN).
  (2) After driving N train_steps that fire actor: counts ==
      (n_updates, n_actor_updates) match; means are finite; alpha
      is in (0, 1] (reasonable bound).
  (3) Drain twice in a row → second drain returns the zero
      snapshot (proves accumulator reset). The values from drain #1
      must NOT leak into drain #2.

Smoke uses synthetic transitions (no env) so the test is fast and
the cadence math is exact."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear

from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents2.redq.kernels import REDQ_TARGET_MIN
from mojo_rl.deep_agents2.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6, OFEPredictorHead,
    REDQOFETrainer,
    state_branch_out_dim, action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 32
comptime CAP = 1_024
comptime PER_UNIT = 4
comptime N_BLOCKS = 6
comptime N = 2
comptime N_MIN = 2
comptime UTD = 1
comptime POLICY_DELAY = 1
comptime WARMUP = 60

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime PRED = OFEPredictorHead[PHI_SA_DIM, OBS]
comptime ACTOR = StochasticActor[
    PHI_S_DIM, ACT,
    Linear[PHI_S_DIM, 32],
]
comptime CRITIC = Sequential[Linear[PHI_SA_DIM, 32], Linear[32, 1]]
comptime SAMPLE = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQOFETrainer[
    "cpu", SAMPLE, ACTOR, CRITIC, SB, AB, PRED,
    N, N_MIN, UTD, POLICY_DELAY, REDQ_TARGET_MIN,
]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_redqofe_metrics_cpu() raises:
    print("=" * 70)
    print("O.3+ — REDQOFEMetrics + flush_metrics (CPU)")
    print("=" * 70)
    seed(42)

    var trainer = Trainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        ofe_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](1.0),
        init_alpha=Scalar[DT](0.2),
        learning_starts=WARMUP,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )

    # (1) Fresh-agent flush: all zero sentinels, no NaN.
    var m0 = trainer.flush_metrics()
    print(
        "  fresh: n_updates=", m0.n_updates,
        " critic_loss=", m0.critic_loss,
        " actor_loss=", m0.actor_loss,
        " aux_loss=", m0.aux_loss,
        " alpha=", m0.alpha,
    )
    assert_true(m0.n_updates == 0, "fresh n_updates must be 0")
    assert_true(m0.n_actor_updates == 0, "fresh n_actor_updates must be 0")
    assert_true(m0.critic_loss == Scalar[DT](0.0), "fresh critic_loss == 0")
    assert_true(m0.actor_loss == Scalar[DT](0.0), "fresh actor_loss == 0")
    assert_true(m0.aux_loss == Scalar[DT](0.0), "fresh aux_loss == 0")
    assert_true(m0.alpha == Scalar[DT](0.0), "fresh alpha == 0")
    assert_true(
        m0.log_prob_mean == m0.log_prob_mean,
        "fresh log_prob_mean must be finite (NaN gate)",
    )

    # (2) Drive synthetic transitions + train_steps.
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    comptime N_DRIVE = 100
    var n_train_step_fired = 0
    for step in range(N_DRIVE):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.2 + 0.005 * Float64(step))
        trainer.select_action(obs, act, step)
        for d in range(OBS):
            nxt[d] = Scalar[DT](0.21 + 0.005 * Float64(step + 1))
        var rew = Scalar[DT](-0.3 + 0.2 * Float64(act[0]))
        var done = (
            Scalar[DT](1.0) if step % 20 == 19 else Scalar[DT](0.0)
        )
        trainer.record(obs, act, rew, nxt, done)
        if done == Scalar[DT](1.0):
            trainer.end_episode()
        if trainer.train_step(step):
            n_train_step_fired += 1

    print(
        "  after", N_DRIVE,
        "env steps -> train_step fired", n_train_step_fired, "times",
    )

    var m1 = trainer.flush_metrics()
    print(
        "  flush#1: n_updates=", m1.n_updates,
        " n_actor=", m1.n_actor_updates,
        " critic_loss=", m1.critic_loss,
        " actor_loss=", m1.actor_loss,
        " aux_loss=", m1.aux_loss,
        " alpha=", m1.alpha,
        " lp_mean=", m1.log_prob_mean,
    )
    assert_true(
        m1.n_updates == n_train_step_fired,
        "n_updates must match train_step-fired count",
    )
    # POLICY_DELAY=1 + UTD=1 → actor fires every train_step.
    assert_true(
        m1.n_actor_updates == n_train_step_fired,
        "n_actor_updates must match (POLICY_DELAY=UTD=1)",
    )
    assert_true(m1.critic_loss == m1.critic_loss, "critic_loss finite")
    assert_true(m1.actor_loss == m1.actor_loss, "actor_loss finite")
    assert_true(m1.aux_loss == m1.aux_loss, "aux_loss finite")
    assert_true(m1.alpha > Scalar[DT](0.0), "alpha must remain positive")
    assert_true(m1.alpha < Scalar[DT](1.0), "alpha must stay bounded")
    assert_true(
        m1.aux_loss > Scalar[DT](0.0),
        "aux_loss mean must be > 0 (Σ MSE divided by count)",
    )

    # (3) Second drain — accumulators were reset → all zero.
    var m2 = trainer.flush_metrics()
    print(
        "  flush#2: n_updates=", m2.n_updates,
        " critic_loss=", m2.critic_loss,
    )
    assert_true(m2.n_updates == 0, "second drain n_updates == 0")
    assert_true(
        m2.critic_loss == Scalar[DT](0.0),
        "second drain critic_loss == 0 (reset gate)",
    )
    assert_true(
        m2.aux_loss == Scalar[DT](0.0),
        "second drain aux_loss == 0 (reset gate)",
    )

    print("PASS — REDQOFEMetrics + flush_metrics drain/reset wired.")


def main() raises:
    test_redqofe_metrics_cpu()

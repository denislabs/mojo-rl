"""O.3+ — REDQOFETrainer.train_step per-inner-tick re-sample gate (UTD>1).

After the per-inner-tick re-sample refactor, `train_step` calls
`sample_blk.step` UTD times per outer call (one before each inner
tick) instead of once per outer call. This test gates:

  (1) At UTD=2, each `train_step` call increments
      `_total_train_steps` by 2 (UTD inner ticks fired).
  (2) `_inner_count` matches N · UTD after N train_step calls.
  (3) Losses are finite throughout, alpha stays bounded.
  (4) The metrics struct still aggregates correctly at UTD>1:
      `n_updates == n_train_step` (one bundle per outer call,
      not per inner tick) — invariant from O.3 (3).

Synthetic transitions only — gates the cadence, not convergence."""

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
comptime UTD = 2
comptime POLICY_DELAY = 2
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


def test_train_step_utd2_per_inner_resample() raises:
    print("=" * 70)
    print("O.3+ — train_step UTD=2 per-inner-tick re-sample gate (CPU)")
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

    # Synthetic transitions — drive enough env steps to clear warmup
    # AND give train_step a runway to fire many times.
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    comptime N_ENV = 130
    var n_train_step_fired = 0
    for step in range(N_ENV):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.2 + 0.005 * Float64(step))
        trainer.select_action(obs, act, step)
        for d in range(OBS):
            nxt[d] = Scalar[DT](0.21 + 0.005 * Float64(step + 1))
        var rew = Scalar[DT](-0.3 + 0.2 * Float64(act[0]))
        var done = (
            Scalar[DT](1.0) if step % 25 == 24 else Scalar[DT](0.0)
        )
        trainer.record(obs, act, rew, nxt, done)
        if done == Scalar[DT](1.0):
            trainer.end_episode()
        if trainer.train_step(step):
            n_train_step_fired += 1

    var ts = trainer.total_train_steps()
    var ic = trainer.inner_count()
    print("  n_train_step_fired       =", n_train_step_fired)
    print("  total_train_steps        =", ts)
    print("  inner_count              =", ic)
    print("  total_train_steps / UTD  =", ts // UTD)

    # (1) Each train_step at UTD=2 must increment total_train_steps
    # by 2 (per-inner-tick re-sample fires twice per outer call).
    assert_true(
        ts == n_train_step_fired * UTD,
        "total_train_steps must equal n_train_step_fired · UTD",
    )
    # (2) inner_count tracks the same.
    assert_true(
        ic == n_train_step_fired * UTD,
        "inner_count must equal n_train_step_fired · UTD",
    )

    # (3) Metrics — one bundle per outer call, regardless of UTD.
    var m = trainer.flush_metrics()
    print(
        "  metrics: n_updates=", m.n_updates,
        " n_actor=", m.n_actor_updates,
        " critic_loss=", m.critic_loss,
        " aux_loss=", m.aux_loss,
    )
    assert_true(
        m.n_updates == n_train_step_fired,
        "n_updates must match n_train_step_fired (one bundle per outer)",
    )
    # POLICY_DELAY=2, UTD=2 → actor fires every 2 inner ticks → once
    # per outer call.
    assert_true(
        m.n_actor_updates == n_train_step_fired,
        "n_actor_updates must equal n_train_step_fired at UTD=POL_DELAY=2",
    )
    assert_true(m.critic_loss == m.critic_loss, "critic_loss finite")
    assert_true(m.aux_loss == m.aux_loss, "aux_loss finite")
    assert_true(
        m.alpha > Scalar[DT](0.0) and m.alpha < Scalar[DT](1.0),
        "alpha must stay in (0, 1)",
    )

    print("PASS — train_step at UTD=2 fires UTD inner ticks per outer call.")


def main() raises:
    test_train_step_utd2_per_inner_resample()

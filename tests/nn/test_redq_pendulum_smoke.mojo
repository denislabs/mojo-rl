"""Phase R.4 — REDQTrainer end-to-end on Pendulum V1 (CPU).

Gates the trait wiring: `REDQTrainer` plugs into the existing
`run_offpolicy_train` (single-env, CPU) driver unchanged. Uses
StochasticActor + Sequential critic to match the SAC Pendulum example
shape so any difference must come from the REDQ algorithm itself,
not the network or env plumbing.

Configuration: N=2, N_MIN=2, UTD=1, POLICY_DELAY=1, MODE=MIN. At
these settings REDQ's TD target reduces to `min(Q0, Q1)` — exactly
the SAC target — but the actor loss is the AVERAGE rather than the
MIN of online critics (the actual REDQ difference). At UTD=1 +
POLICY_DELAY=1 the schedule also matches SAC, so this is a "REDQ
in SAC's clothes" smoke that mainly validates the driver wiring.

Budget: 10_000 env steps. SAC needs ~30k for swing-up (-167) on
Pendulum; we just want the trainer to (a) run end-to-end without
crashing and (b) clearly improve over the random baseline (-1250)
within the time budget.

Gates:
  (a) Driver returns episode-return list (non-empty).
  (b) Trainer completed at least one episode (ep_count > 0).
  (c) total_train_steps == (TOTAL_TIMESTEPS - learning_starts) ·
      UTD — the warmup gate fired UTD inner steps per non-warmup
      env step.
  (d) Final mean_return is finite and STRICTLY BETTER than the
      -1250 random-policy baseline.
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU

from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.training.driver_offpolicy import run_offpolicy_train
from mojo_rl.deep_agents.redq import REDQTrainer, REDQ_TARGET_MIN

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime CAP = 50_000
comptime TOTAL_TIMESTEPS = 10_000
comptime WARMUP = 1_000

comptime N = 2
comptime N_MIN = 2
comptime UTD = 1
comptime POLICY_DELAY = 1
comptime Q_MODE = REDQ_TARGET_MIN

comptime ActorNet = StochasticActor[
    OBS, ACT,
    Linear[OBS, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Sample = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQTrainer[
    "cpu", Sample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]


def test_redq_pendulum_smoke() raises:
    print("=" * 70)
    print(
        "R.4 — REDQ N=2 M=2 UTD=1 POL_DELAY=1 MIN on Pendulum V1 (CPU)"
    )
    print("=" * 70)
    seed(42)

    var trainer = Trainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=WARMUP,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var ep_returns = run_offpolicy_train[Trainer, PendulumEnv[DT]](
        trainer,
        env,
        TOTAL_TIMESTEPS,
        ctx=None,
        print_every=2_000,
        verbose=True,
    )

    var final_mean = trainer.mean_return()
    var ep = trainer.ep_count()
    var ts = trainer.total_train_steps()
    print("Final mean ep return (last 10):", final_mean)
    print("Episodes completed:            ", ep)
    print("Total inner train steps:       ", ts)
    print("ep_returns list length:        ", len(ep_returns))

    # (a) Driver completed and produced returns.
    assert_true(
        len(ep_returns) > 0,
        "driver must return at least one completed-episode return",
    )
    # (b) The trainer's tracker saw episodes.
    assert_true(ep > 0, "trainer must complete at least one episode")
    # (c) Inner-step accounting. After warmup every env-loop iteration
    # runs UTD inner updates. The driver passes step_idx = 1..TOTAL
    # (it post-increments before `train_step` — driver_offpolicy.mojo:416),
    # so the warmup gate `step_idx < learning_starts` admits step_idx ∈
    # [WARMUP, TOTAL] → `TOTAL − WARMUP + 1` outer calls × UTD inner
    # updates each.
    var expected_ts = (TOTAL_TIMESTEPS - WARMUP + 1) * UTD
    assert_true(
        ts == expected_ts,
        "total_train_steps must equal (TOTAL − WARMUP + 1) · UTD",
    )
    # (d) Mean return finite and clearly above baseline (-1250).
    var fr = Float64(final_mean)
    assert_true(fr == fr, "final mean_return finite")
    assert_true(
        fr > -1200.0,
        "REDQ must improve over the -1250 random baseline within 10k steps",
    )

    if fr > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif fr > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif fr > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — modest improvement, still exploring.")

    print("PASS — REDQ Pendulum smoke green.")


def main() raises:
    test_redq_pendulum_smoke()

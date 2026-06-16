"""O.2.b.4 — REDQOFETrainer end-to-end on Pendulum V1 (CPU).

Runs a manual env loop on Pendulum (skipping `run_offpolicy_train`
since REDQOFETrainer doesn't yet conform to the full `OffPolicyAgent`
trait — that lands in a follow-up slice). The loop calls:

    trainer.select_action(obs, action, step)
    env.step_continuous_vec(action) → (next_obs, reward, done)
    trainer.record(obs, action, reward, next_obs, done)
    if done: trainer.end_episode(); env.reset()
    if step >= WARMUP: trainer.train_step(step)

Gates:
  (a) End-to-end loop completes without crashing.
  (b) Trainer completed at least one episode (ep_count > 0).
  (c) total_train_steps == (TOTAL − WARMUP) · UTD.
  (d) mean_return is finite AND beats the −1250 random baseline by
      a comfortable margin. We use a budget too small to fully solve
      Pendulum (REDQ-OFE shines at much longer horizons + Ant /
      Humanoid scale); the gate is "clearly learning", not "solved".

Configuration: smaller-scale REDQ-OFE — N=2, N_MIN=2, UTD=1,
POLICY_DELAY=1, 6-block branches, per_unit=8 (φ width = OBS+48 = 51).
At UTD=1 + POLICY_DELAY=1 the actor cadence matches SAC; the
trainable difference vs SAC is the OFE feature pre-pass + aux loss
on every train step."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU

from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.redq.kernels import REDQ_TARGET_MIN
from mojo_rl.deep_agents.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6, OFEPredictorHead,
    REDQOFETrainer,
    state_branch_out_dim, action_branch_out_dim,
)

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 128
comptime CAP = 20_000
comptime PER_UNIT = 8
comptime N_BLOCKS = 6
comptime N = 2
comptime N_MIN = 2
comptime UTD = 1
comptime POLICY_DELAY = 1
comptime TOTAL_TIMESTEPS = 5_000
comptime WARMUP = 500

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)   # 51
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

# Network types.
comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime PRED = OFEPredictorHead[PHI_SA_DIM, OBS]
comptime ACTOR = StochasticActor[
    PHI_S_DIM, ACT,
    Linear[PHI_S_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CRITIC = Sequential[
    Linear[PHI_SA_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime SAMPLE = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]


def test_redq_ofe_pendulum_smoke() raises:
    print("=" * 70)
    print("O.2.b.4 — REDQ-OFE N=2 M=2 UTD=1 6-block on Pendulum V1 (CPU)")
    print("=" * 70)
    seed(42)

    var trainer = REDQOFETrainer[
        "cpu", SAMPLE, ACTOR, CRITIC, SB, AB, PRED,
        N, N_MIN, UTD, POLICY_DELAY, REDQ_TARGET_MIN,
    ].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        ofe_lr=Scalar[DT](3e-4),
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

    # ── Env loop ──────────────────────────────────────────────────────
    _ = env.reset()
    var obs = env.get_obs_list()
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    for step in range(TOTAL_TIMESTEPS):
        trainer.select_action(obs, action, step)
        var result = env.step_continuous_vec[DT](action)
        var next_obs = result[0].copy()
        var reward = result[1]
        var done = result[2]
        var done_f = Scalar[DT](1.0) if done else Scalar[DT](0.0)
        trainer.record(obs, action, reward, next_obs, done_f)
        obs = next_obs.copy()
        if done:
            trainer.end_episode()
            _ = env.reset()
            obs = env.get_obs_list()
        if step >= WARMUP:
            _ = trainer.train_step(step)
        if step > 0 and step % 1000 == 0:
            print(
                "  step", step,
                " mean_ret(last 10)=", trainer.mean_return(),
                " ep_count=", trainer.ep_count(),
                " train_steps=", trainer.total_train_steps(),
            )

    var final_mean = trainer.mean_return()
    var ep = trainer.ep_count()
    var ts = trainer.total_train_steps()
    print("Final mean ep return (last 10):", final_mean)
    print("Episodes completed:            ", ep)
    print("Total inner train steps:       ", ts)

    # ── Gates ─────────────────────────────────────────────────────────
    # (b) Trainer completed at least one episode.
    assert_true(ep > 0, "trainer must complete at least one episode")

    # (c) Inner-step accounting. Our train_step is called for every
    # env step in [WARMUP, TOTAL) → (TOTAL − WARMUP) outer calls ×
    # UTD inner updates each.
    var expected_ts = (TOTAL_TIMESTEPS - WARMUP) * UTD
    assert_true(
        ts == expected_ts,
        "total_train_steps must equal (TOTAL − WARMUP) · UTD",
    )

    # (d) Mean return finite and BEATS the random baseline (-1250).
    var fr = Float64(final_mean)
    assert_true(fr == fr, "final mean_return finite")
    assert_true(
        fr > -1200.0,
        "REDQ-OFE must clearly improve over the −1250 random baseline",
    )

    if fr > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif fr > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif fr > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — modest improvement, still exploring.")
    print("PASS — REDQ-OFE Pendulum smoke green.")


def main() raises:
    test_redq_ofe_pendulum_smoke()

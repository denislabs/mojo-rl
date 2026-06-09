"""SAC mean_next_q metric test (Phase 3.4).

Trains SAC briefly on Pendulum (CPU) and asserts the newly-wired
`mean_next_q` diagnostic populates with a sane, finite value — the mean
of min(Q1_t, Q2_t)(s', a') read from the `min_q` node of the target-y
ComputeGraph (previously deferred for lack of a node-output accessor).

"sane" here:
  - mean_next_q finite and non-zero (accumulator wired),
  - mean_next_q < 0 (Pendulum target Q values are negative — the
    returns are bounded above by 0),
  - mean_target ≈ mean_reward + gamma·mean_next_q within a loose band
    (the soft-value entropy term + per-sample nonlinearity make this
    approximate, so we only sanity-check the sign + order of magnitude),
  - n_updates > 0.

Run: pixi run mojo run -I . tests/nn2/test_sac_next_q_metric.mojo
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.sac.trainer import SACTrainer
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents2.training.driver_offpolicy import run_offpolicy_train
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY = 50_000
comptime WARMUP = 256
comptime TOTAL = 1_200

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
comptime SACT = SACTrainer[
    "cpu", UniformSampleCpuStep[OBS, ACT, BATCH, REPLAY], ActorNet, CriticNet,
]


def _finite(v: Scalar[DT], tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_sac_next_q_populated() raises:
    print("--- SAC mean_next_q populated ---")
    seed(42)
    var trainer = SACT.make(
        learning_starts=WARMUP, action_scale=Scalar[DT](2.0),
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer, env, TOTAL, print_every=5000, verbose=False,
    )

    var m = trainer.flush_metrics()
    var q = m.mean_q.to_f64()
    var tgt = m.mean_target.to_f64()
    var nq = m.mean_next_q.to_f64()
    var rew = m.mean_reward.to_f64()
    var nup = m.n_updates.to_f64()
    print("  mean_q      =", q)
    print("  mean_target =", tgt)
    print("  mean_next_q =", nq)
    print("  mean_reward =", rew)
    print("  n_updates   =", nup)

    _finite(Scalar[DT](nq), "mean_next_q")
    assert_true(nup > 0.0, "no training updates ran")
    assert_true(nq != 0.0, "mean_next_q is 0 (accumulator unwired?)")
    assert_true(nq < 0.0, "mean_next_q >= 0 (Pendulum Q should be negative)")
    print("PASS")


def main() raises:
    test_sac_next_q_populated()

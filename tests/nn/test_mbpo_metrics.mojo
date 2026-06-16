"""MBPO metrics-parity test (Phase 3.4).

Trains MBPO briefly on Pendulum (CPU) and asserts the new learning
diagnostics (`mean_q`, `mean_reward`, `dyn_loss`) populate with sane,
finite values — not left at zero.

"sane" here:
  - all fields finite,
  - n_updates > 0 (SAC inner updates ran),
  - dyn_loss > 0 (dynamics ensemble trained at least one member-step),
  - mean_reward < 0 (Pendulum reward is in [-16.27, 0]),
  - mean_q finite (no support bound on a scalar critic).

A 0.0 on `dyn_loss` catches an unwired dynamics accumulator; a 0.0 on
`mean_reward` catches an unwired diag walk.

Run: pixi run mojo run -I . tests/nn/test_mbpo_metrics.mojo
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.mbpo.trainer import MBPOTrainer
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime DYN_HIDDEN = 64
comptime BATCH = 64
comptime REPLAY_CAP = 10_000
comptime SYNTH_CAP = 20_000
comptime N_ENS = 4
comptime N_ELITES = 3

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
comptime DynNet = Sequential[
    Linear[OBS + ACT, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS)],
]
comptime Trainer = MBPOTrainer[
    "cpu", ActorNet, CriticNet, DynNet,
    OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, 5,
]


def _finite(v: Scalar[DT], tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_mbpo_metrics_populated() raises:
    print("--- MBPO metrics populated ---")
    seed(42)
    var t = Trainer.make(
        action_scale=Scalar[DT](2.0),
        learning_starts=200,
        model_train_freq=200,
        num_rollouts_per_step=64,
        sac_updates_per_step=4,
        dyn_epochs_per_round=1,
    )
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs = env.get_obs_list()
    var action = List[Scalar[DT]](capacity=ACT)
    action.append(Scalar[DT](0.0))

    var max_steps = 600
    for step_idx in range(max_steps):
        t.select_action(obs, action, step_idx)
        var step_res = env.step_continuous(action[0])
        var next_obs = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        t.record(
            obs, action, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            t.end_episode()
            _ = env.reset()
            obs = env.get_obs_list()
        else:
            obs = next_obs.copy()
        _ = t.train_step(step_idx)

    var m = t.flush_metrics()
    var aq = m.actor_loss.to_f64()
    var cq = m.critic_loss.to_f64()
    var alpha = m.alpha.to_f64()
    var q = m.mean_q.to_f64()
    var rew = m.mean_reward.to_f64()
    var dl = m.dyn_loss.to_f64()
    var ts = m.train_steps.to_f64()
    var nup = m.n_updates.to_f64()
    print("  actor_loss  =", aq)
    print("  critic_loss =", cq)
    print("  alpha       =", alpha)
    print("  mean_q      =", q)
    print("  mean_reward =", rew)
    print("  dyn_loss    =", dl)
    print("  train_steps =", ts)
    print("  n_updates   =", nup)

    _finite(Scalar[DT](aq), "actor_loss")
    _finite(Scalar[DT](cq), "critic_loss")
    _finite(Scalar[DT](alpha), "alpha")
    _finite(Scalar[DT](q), "mean_q")
    _finite(Scalar[DT](rew), "mean_reward")
    _finite(Scalar[DT](dl), "dyn_loss")

    assert_true(nup > 0.0, "no SAC inner updates ran")
    assert_true(ts > 0.0, "cumulative train_steps is 0")
    assert_true(dl > 0.0, "dyn_loss is 0 (dynamics accumulator unwired?)")
    assert_true(
        rew < 0.0, "mean_reward >= 0 (Pendulum reward should be negative)"
    )
    assert_true(alpha > 0.0, "alpha should be positive (= exp(log_alpha))")
    print("PASS")


def main() raises:
    test_mbpo_metrics_populated()

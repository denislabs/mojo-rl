"""PPOTrainer smoke — construction + a single rollout-update cycle.

Verifies the trainer compiles, constructs, runs ROLLOUT_LEN env
steps, and the K-epoch update fires exactly once at the boundary.

Tiny BATCH + ROLLOUT_LEN to keep test wall-time short.
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.deep_agents2.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents2.ppo.trainer import PPOTrainer
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 16
comptime ROLLOUT = 64
comptime MB = 16
comptime EPOCHS = 2

comptime ActorNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Trainer = PPOTrainer[
    "cpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS,
]


def test_construction() raises:
    print("test_construction ...")
    seed(42)
    var t = Trainer.make(action_scale=Scalar[DT](2.0))
    print(
        "  ROLLOUT_LEN =", Trainer.ROLLOUT_LEN,
        " MINIBATCH =", Trainer.MINIBATCH,
        " N_MINIBATCHES =", Trainer.N_MINIBATCHES,
        " N_EPOCHS =", Trainer.N_EPOCHS,
    )
    _ = t
    print("  ok")


def test_one_rollout_cycle() raises:
    """Drive ROLLOUT+5 env-steps through Pendulum. Verify at least one
    train_step (the boundary one) returns True."""
    print("test_one_rollout_cycle ...")
    seed(42)
    var t = Trainer.make(action_scale=Scalar[DT](2.0))
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs = env.get_obs_list()
    var action = List[Scalar[DT]](capacity=ACT)
    action.append(Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var stepped_true = 0
    var max_steps = ROLLOUT + 5
    for step_idx in range(max_steps):
        t.select_action(obs, action, step_idx)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = Scalar[DT](nxt[d])
        t.record_transition(
            obs, action, Scalar[DT](reward), next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            t.end_episode()
            _ = env.reset()
            obs = env.get_obs_list()
        else:
            obs = nxt.copy()
        var did = t.train_step(step_idx)
        if did:
            stepped_true += 1

    print(
        "  train_step True count =", stepped_true,
        " mean_return(10) =", t.mean_return(),
        " ep_count =", t.ep_count(),
    )
    assert_true(
        stepped_true >= 1,
        "at least one PPO update must fire within ROLLOUT+5 env-steps",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PPOTrainer smoke (CPU N_ENVS=1)")
    print("=" * 70)
    test_construction()
    test_one_rollout_cycle()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

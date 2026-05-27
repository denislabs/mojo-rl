"""MBPOTrainer smoke test (Step 4)."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.mbpo_trainer import MBPOTrainer
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
    ActorNet, CriticNet, DynNet,
    OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, 5,
]


def test_end_to_end_few_steps() raises:
    print("test_end_to_end_few_steps ...")
    seed(42)
    var t = Trainer.make["cpu"](
        action_scale=Scalar[DT](2.0),
        learning_starts=200,
        model_train_freq=200,
        num_rollouts_per_step=64,
        sac_updates_per_step=2,
        dyn_epochs_per_round=1,
    )
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs = env.get_obs_list()
    var action = List[Scalar[DT]](capacity=ACT)
    action.append(Scalar[DT](0.0))

    var stepped_count = 0
    var max_steps = 410
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

        var did = t.train_step(step_idx)
        if did:
            stepped_count += 1

    print("  total train steps executed:", stepped_count)
    print("  synth buf size:", t.sample_blk.synth_buf.size)
    print("  real buf size:", t.sample_blk.real_buf.size)
    print("  mean_return:", t.mean_return())
    assert_true(
        t.sample_blk.synth_buf.size > 0,
        "synth buffer should be populated",
    )
    assert_true(
        stepped_count > 0,
        "at least one MBPO train_step should have executed post-warmup",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Step 4 — MBPOTrainer smoke")
    print("=" * 70)
    test_end_to_end_few_steps()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

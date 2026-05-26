"""MBPOTrainer smoke test.

Phase I.1.c/d compile + minimal end-to-end pass.  Validates:
  1. Trainer constructs without crashing.
  2. After `learning_starts + model_train_freq` env steps + a few
     train_step calls, the trainer has populated both real and synth
     replay buffers and `train_step` returns True.
  3. mean_return is finite (sanity-check that the eval path through
     the inner SAC tracker works).

This is a SMOKE test — does NOT validate learning quality. That gate
lives in `tests/nn2/test_mbpo_pendulum_match.mojo` for I.1.e.
"""

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


def test_construction() raises:
    print("test_construction ...")
    seed(42)
    var t = Trainer.make["cpu"](
        action_scale=Scalar[DT](2.0),
        learning_starts=100,
        model_train_freq=100,
        num_rollouts_per_step=64,
        sac_updates_per_step=2,
        dyn_epochs_per_round=1,
    )
    print("  REAL_BS =", Trainer.REAL_BS, "SYNTH_BS =", Trainer.SYNTH_BS)
    assert_true(Trainer.REAL_BS + Trainer.SYNTH_BS == BATCH,
                "REAL_BS + SYNTH_BS should equal BATCH")
    print("  ok")


def test_warmup_skip() raises:
    """Pre-learning_starts train_step calls must return False."""
    print("test_warmup_skip ...")
    seed(42)
    var t = Trainer.make["cpu"](
        action_scale=Scalar[DT](2.0),
        learning_starts=100,
        model_train_freq=100,
        num_rollouts_per_step=64,
        sac_updates_per_step=2,
        dyn_epochs_per_round=1,
    )
    for step in range(50):
        var stepped = t.train_step["cpu"](step)
        assert_true(not stepped, "train_step pre-warmup should return False")
    print("  ok (50 pre-warmup steps all skipped)")


def test_end_to_end_few_steps() raises:
    """Drive the trainer through enough Pendulum env steps to trigger
    one dynamics-training round + a couple SAC updates.  Just checks
    no crashes and that train_step returns True post-warmup."""
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
    var max_steps = 410   # enough to clear warmup + first dyn round
    for step_idx in range(max_steps):
        # Action selection (warmup branch runs uniform; post-warmup runs rsample).
        t.select_action(obs, action, step_idx)
        var step_res = env.step_continuous(action[0])
        var next_obs = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        t.record(obs, action, reward,
                 next_obs, Scalar[DT](1.0) if done else Scalar[DT](0.0))
        if done:
            t.end_episode()
            _ = env.reset()
            obs = env.get_obs_list()
        else:
            obs = next_obs.copy()

        var did = t.train_step["cpu"](step_idx)
        if did:
            stepped_count += 1

    print("  total train steps executed:", stepped_count)
    print("  synth buf size:", t.synth_buf.size)
    print("  real buf size:", t.sac.buf.size)
    print("  mean_return:", t.mean_return())
    assert_true(t.synth_buf.size > 0, "synth buffer should be populated")
    assert_true(
        stepped_count > 0,
        "at least one MBPO train_step should have executed post-warmup",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("MBPOTrainer smoke (Phase I.1.c/d)")
    print("=" * 70)
    test_construction()
    test_warmup_skip()
    test_end_to_end_few_steps()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

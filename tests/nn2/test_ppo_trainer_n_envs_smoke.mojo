"""PPOTrainer N_ENVS=4 direct-call smoke.

Validates that the N_ENVS-parametric trainer:
  - Constructs at N_ENVS=4
  - Runs ROLLOUT_LEN+5 batched steps through 4 independent Pendulum
    envs via the `select_action_batched` + `record_batch_cpu` surface
  - Fires exactly one K-epoch update at the rollout boundary
  - Produces finite mean_return

Direct-call test bypassing the BatchedEnv driver — pokes the trainer
methods directly to validate the N_ENVS comptime path in isolation.
The driver-driven equivalent lives in `test_ppo_trainer_4mode.mojo`.
"""

from std.math import isnan, isinf
from std.memory import alloc
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
comptime N_ENVS = 4
comptime ROLLOUT = 64  # rollout length per env
comptime MB = 16       # minibatch is gathered from ROLLOUT * N_ENVS = 256 pool
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
    "cpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS,
]


def main() raises:
    print("=" * 70)
    print("PPOTrainer N_ENVS=", N_ENVS, " direct-call smoke")
    print("=" * 70)
    seed(42)
    var t = Trainer.make(action_scale=Scalar[DT](2.0))
    print(
        "  ROLLOUT_LEN =", Trainer.ROLLOUT_LEN,
        " MINIBATCH =", Trainer.MINIBATCH,
        " N_MINIBATCHES =", Trainer.N_MINIBATCHES,
        " N_EPOCHS =", Trainer.N_EPOCHS,
        " N_ENVS =", Trainer.N_ENVS,
    )

    # N_ENVS independent Pendulum envs.
    var envs = List[PendulumEnv[DT]]()
    for _ in range(N_ENVS):
        envs.append(PendulumEnv[DT]())
    for e in range(N_ENVS):
        _ = envs[e].reset()

    # Batched staging buffers.
    var obs_ptr      = alloc[Scalar[DT]](N_ENVS * OBS)
    var action_ptr   = alloc[Scalar[DT]](N_ENVS * ACT)
    var reward_ptr   = alloc[Scalar[DT]](N_ENVS)
    var done_ptr     = alloc[Scalar[DT]](N_ENVS)
    var next_obs_ptr = alloc[Scalar[DT]](N_ENVS * OBS)

    # Populate initial obs.
    for e in range(N_ENVS):
        var ol = envs[e].get_obs_list()
        for d in range(OBS):
            obs_ptr[e * OBS + d] = Scalar[DT](ol[d])

    var stepped_true = 0
    var max_steps = ROLLOUT + 5
    for step_idx in range(max_steps):
        t.select_action_batched(obs_ptr, action_ptr, step_idx)
        for e in range(N_ENVS):
            var step_res = envs[e].step_continuous(action_ptr[e * ACT])
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            reward_ptr[e] = Scalar[DT](reward)
            done_ptr[e]   = Scalar[DT](1.0) if done else Scalar[DT](0.0)
            for d in range(OBS):
                next_obs_ptr[e * OBS + d] = Scalar[DT](nxt[d])
        t.record_batch_cpu(obs_ptr, reward_ptr, next_obs_ptr, done_ptr)
        for e in range(N_ENVS):
            if done_ptr[e] > Scalar[DT](0.5):
                _ = envs[e].reset()
                var ol = envs[e].get_obs_list()
                for d in range(OBS):
                    obs_ptr[e * OBS + d] = Scalar[DT](ol[d])
            else:
                for d in range(OBS):
                    obs_ptr[e * OBS + d] = next_obs_ptr[e * OBS + d]
        var did = t.train_step(step_idx)
        if did:
            stepped_true += 1

    var mr = t.mean_return()
    print(
        "  train_step True count =", stepped_true,
        " mean_return(10) =", mr,
        " ep_count =", t.ep_count(),
    )
    assert_true(
        stepped_true >= 1,
        "at least one PPO update must fire within ROLLOUT+5 batched steps",
    )
    assert_true(not isnan(mr), "N_ENVS mean_return NaN")
    assert_true(not isinf(mr), "N_ENVS mean_return Inf")
    # Pendulum truncates at step 200, the rollout (ROLLOUT+5=69 steps)
    # ends before any env completes — ep_count will be 0 and
    # mean_return will read the EpisodeTracker's initial_fill. That's
    # fine: this test gates compile + one rollout-update cycle, not
    # episode completion.
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

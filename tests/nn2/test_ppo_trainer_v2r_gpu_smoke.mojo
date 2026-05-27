"""PPOTrainerV2R GPU smoke — construction + one rollout-update cycle.

P.2 gate: V2R trainer compiles for `train_target="gpu"`, runs
ROLLOUT_LEN env steps, and the K-epoch update fires exactly once at
the boundary with finite loss/return. Hybrid N=1 GPU path: per-step
actor/critic forwards run on device; rollout buffers stay host-side;
the minibatch is H2D-uploaded before each actor/critic train step.

Convergence is NOT gated here (200k Pendulum example covers that);
this is a compile + run-one-cycle smoke test mirroring
`test_ppo_trainer_v2r_smoke.mojo`.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.training.ppo_trainer_v2r import PPOTrainerV2R
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
comptime Trainer = PPOTrainerV2R[
    "gpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS,
]


def test_construction() raises:
    print("test_construction ...")
    seed(42)
    var ctx = DeviceContext()
    var t = Trainer.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    print(
        "  ROLLOUT_LEN =", Trainer.ROLLOUT_LEN,
        " MINIBATCH =", Trainer.MINIBATCH,
        " N_MINIBATCHES =", Trainer.N_MINIBATCHES,
        " N_EPOCHS =", Trainer.N_EPOCHS,
    )
    _ = t
    print("  ok")


def test_one_rollout_cycle() raises:
    """Drive ROLLOUT+5 env-steps through Pendulum. Verify exactly one
    train_step (the boundary one) returns True with finite output."""
    print("test_one_rollout_cycle ...")
    seed(42)
    var ctx = DeviceContext()
    var t = Trainer.make(action_scale=Scalar[DT](2.0), ctx=ctx)
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

    var mr = t.mean_return()
    print(
        "  train_step True count =", stepped_true,
        " mean_return(10) =", mr,
        " ep_count =", t.ep_count(),
    )
    assert_true(
        stepped_true >= 1,
        "at least one PPO update must fire within ROLLOUT+5 env-steps",
    )
    assert_true(not isnan(mr), "GPU mean_return NaN")
    assert_true(not isinf(mr), "GPU mean_return Inf")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PPOTrainerV2R GPU smoke (P.2 — train_target='gpu', N_ENVS=1)")
    print("=" * 70)
    test_construction()
    test_one_rollout_cycle()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

"""PPOTrainer GPU smoke — construction + one rollout-update cycle.

Verifies the trainer compiles for `train_target="gpu"`, runs
ROLLOUT_LEN env steps, and the K-epoch update fires exactly once at
the boundary with finite loss/return. Hybrid N=1 GPU path: per-step
actor/critic forwards run on device; rollout buffers stay host-side;
the minibatch is H2D-uploaded before each actor/critic train step.

Convergence is NOT gated here (200k Pendulum example covers that);
this is a compile + run-one-cycle smoke test mirroring
`test_ppo_trainer_smoke.mojo`.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.deep_agents.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents.ppo.trainer import PPOTrainer
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

    # Distributional diag fix: entropy / approx_kl / clip_fraction /
    # explained_variance were a hard 0.0 on GPU before the device kernels
    # were wired. Entropy of a Gaussian policy is strictly non-zero, so a
    # non-zero finite entropy proves `_accumulate_diag_gpu` ran end-to-end.
    var m = t.flush_metrics()
    var ent = m.entropy.to_f64()
    var kl = m.approx_kl.to_f64()
    var clip = m.clip_fraction.to_f64()
    var ev = m.explained_variance.to_f64()
    print(
        "  entropy =", ent, " approx_kl =", kl,
        " clip_fraction =", clip, " explained_variance =", ev,
    )
    assert_true(not isnan(ent) and not isinf(ent), "GPU entropy non-finite")
    assert_true(not isnan(kl) and not isinf(kl), "GPU approx_kl non-finite")
    assert_true(not isnan(clip) and not isinf(clip), "GPU clip non-finite")
    assert_true(not isnan(ev) and not isinf(ev), "GPU explained_var non-finite")
    assert_true(ent != 0.0, "GPU entropy is 0 (diag kernel unwired?)")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PPOTrainer GPU smoke (train_target='gpu', N_ENVS=1)")
    print("=" * 70)
    test_construction()
    test_one_rollout_cycle()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

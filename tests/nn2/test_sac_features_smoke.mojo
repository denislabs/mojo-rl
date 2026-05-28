"""Kwarg smoke for bf16/AMP + ERE features.

PER + n-step are covered by test_sac_trainer_smoke.mojo and
test_sac_nstep_smoke.mojo respectively. This file fills the gap
for use_bf16 (Bf16Compute auto-routing) and use_ere (recency-biased
sampling). Both are GPU-only.

Replaces the legacy test_trainer_auto_bf16.mojo + test_trainer_auto_ere.mojo
which exercised SACConfig → trainer wiring; SACTrainer uses direct kwargs so
the round-trip layer is gone.
"""

from std.gpu.host import DeviceContext
from std.math import isfinite
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.sac.trainer import SACTrainer
from mojo_rl.deep_agents2.training.blocks import UniformSampleGpuStep

from mojo_rl.envs.pendulum import PendulumEnv

comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 5_000
comptime SMOKE_STEPS = 2_000

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]

comptime Trainer = SACTrainer[
    "gpu",
    UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]


def _train_2k(mut trainer: Trainer) raises -> Scalar[DT]:
    var env = PendulumEnv[DT]()
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    _ = env.reset()
    var obs_self = env.get_obs_list()
    for step in range(SMOKE_STEPS):
        for d in range(OBS_DIM):
            obs[d] = obs_self[d]
        trainer.select_action(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS_DIM):
            next_obs[d] = nxt[d]
        trainer.record(
            obs,
            action,
            reward,
            next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
        _ = trainer.train_step(step)
    return trainer.mean_return()


def test_use_bf16_kwarg() raises:
    print("--- use_bf16=True smoke ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
        use_bf16=True,
    )
    assert_true(trainer._use_bf16, "use_bf16 kwarg not stored")
    var mr = _train_2k(trainer)
    print("  mean_return:", mr)
    assert_true(isfinite(mr), "bf16 path produced non-finite mean_return")
    assert_true(
        mr < Scalar[DT](0.0) and mr > Scalar[DT](-2000.0),
        "bf16 mean_return outside plausible Pendulum range: " + String(mr),
    )
    print("  test_use_bf16_kwarg PASSED")


def test_use_ere_kwarg() raises:
    print("--- use_ere=True smoke ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](2.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
        use_ere=True,
        ere_eta=Scalar[DT](0.996),
        ere_c_min=256,
        ere_k_max=4,
    )
    var mr = _train_2k(trainer)
    print("  mean_return:", mr)
    assert_true(isfinite(mr), "ERE path produced non-finite mean_return")
    assert_true(
        mr < Scalar[DT](0.0) and mr > Scalar[DT](-2000.0),
        "ERE mean_return outside plausible Pendulum range: " + String(mr),
    )
    print("  test_use_ere_kwarg PASSED")


def main() raises:
    print("=" * 60)
    print("SAC feature kwarg smoke (bf16 + ERE)")
    print("=" * 60)
    test_use_bf16_kwarg()
    test_use_ere_kwarg()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)

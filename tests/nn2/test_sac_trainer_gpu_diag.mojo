"""SACTrainer GPU diag-metric test.

Drives the GPU SAC trainer (cpu env + gpu train, single-env) on Pendulum for
a short horizon and asserts the device-resident `mean_q` / `mean_reward`
diagnostics populate. Before they were wired, both read a hard 0.0 on GPU
(the per-batch diag walk was CPU-only); this test guards the device-side
reduction added in `_train_post_sample_kernels` + `flush_metrics`.

"sane" here:
  - mean_q / mean_reward finite,
  - mean_reward < 0 (Pendulum reward is a cost — strictly negative),
  - mean_q != 0 (the critic has trained; an exact 0 means the device
    accumulator never folded `mb_q` in),
  - n_updates > 0.

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn2/test_sac_trainer_gpu_diag.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
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
    "gpu", UniformSampleGpuStep[OBS, ACT, BATCH, REPLAY], ActorNet, CriticNet,
]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_sac_gpu_diag_populated() raises:
    print("--- SAC GPU mean_q / mean_reward populated ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACT.make(
        ctx=ctx, learning_starts=WARMUP, action_scale=Scalar[DT](2.0),
    )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs_self = env.get_obs_list()

    var step: Int = 0
    while step < TOTAL:
        for d in range(OBS):
            obs[d] = obs_self[d]
        trainer.select_action(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = nxt[d]
        trainer.record(
            obs, action, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
        step += 1
        _ = trainer.train_step(step)

    var m = trainer.flush_metrics()
    var mq = m.mean_q.to_f64()
    var mr = m.mean_reward.to_f64()
    var nup = m.n_updates.to_f64()
    print("  mean_q      =", mq)
    print("  mean_reward =", mr)
    print("  n_updates   =", nup)

    _finite(mq, "mean_q")
    _finite(mr, "mean_reward")
    assert_true(nup > 0.0, "no training updates ran")
    assert_true(mr < 0.0, "mean_reward should be < 0 on Pendulum (device diag)")
    assert_true(mq != 0.0, "mean_q is 0 on GPU (device accumulator unwired?)")
    print("PASS")


def main() raises:
    print("=" * 70)
    print("SAC GPU diag-metric test")
    print("=" * 70)
    test_sac_gpu_diag_populated()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

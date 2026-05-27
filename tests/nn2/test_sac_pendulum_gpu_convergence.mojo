"""SAC Pendulum GPU convergence regression (Phase 0.3).

Runs the GPU SAC trainer against the real Pendulum env for 30k steps and
asserts `mean10 < -200`. This locks in the end-to-end GPU SAC path that
Block A + Block D unblocked; previously only a 36-step NaN smoke
(`test_sac_pendulum_gpu.mojo`) covered the GPU surface.

Reference (CPU baseline): mean10 = -170.2601 at 30k. GPU mean10 differs
because the GPU box_muller uses Philox vs CPU's std.random; convergence
target on GPU is the `>-200 EXCELLENT` threshold (consistently hit at
~-121 on Apple Silicon during Block D validation).

Run:
    pixi run mojo run -I . tests/nn2/test_sac_pendulum_gpu_convergence.mojo
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer_v2r import SACTrainerV2R
from mojo_rl.nn2.training.blocks_ref import UniformSampleGpuStep

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TOTAL_TIMESTEPS = 30_000
comptime CONVERGENCE_THRESHOLD = -200.0  # GPU SAC must beat random baseline by a margin

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def test_sac_pendulum_gpu_convergence() raises:
    seed(42)
    var ctx = DeviceContext()
    var trainer = SACTrainerV2R[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
    )

    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs_self = env.get_obs_list()

    var t_start = perf_counter_ns()
    var step: Int = 0
    while step < TOTAL_TIMESTEPS:
        for d in range(OBS_DIM):
            obs[d] = obs_self[d]
        trainer.select_action_gpu(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS_DIM):
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
        _ = trainer.train_step_gpu(step)

        if step % 5_000 == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ", step, "] mean_ret(10)=", trainer.mean_return(),
                " ep=", trainer.ep_count(),
                " elapsed=", elapsed, "s",
            )

    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
    var final_mean = Float64(trainer.mean_return())
    print("=" * 70)
    print("Final mean10 =", final_mean, " (threshold:", CONVERGENCE_THRESHOLD, ")")
    print("Total wall-time =", elapsed, "s")
    print(trainer.flush_timer_log())
    print("=" * 70)
    assert_true(
        final_mean > CONVERGENCE_THRESHOLD,
        "GPU SAC failed convergence regression: mean10 not above threshold"
    )
    print("  test_sac_pendulum_gpu_convergence PASSED")


def main() raises:
    print("=" * 70)
    print("SAC Pendulum GPU convergence regression (Phase 0.3)")
    print("=" * 70)
    test_sac_pendulum_gpu_convergence()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

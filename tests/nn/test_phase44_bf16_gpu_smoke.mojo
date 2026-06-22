"""Phase 4.4 bf16/AMP GPU smoke — DDPG + TD3 + MBPO.

Builds each GPU trainer with `use_bf16=True` and runs a short Pendulum
loop (cpu env + gpu train), asserting the train step runs under the
`Bf16Compute` AMP policy and drains finite metrics. This locks in the
bf16 plumbing threaded through the device train step (mixed-precision
forward/vjp on the critic / actor / target-y matmuls).

NoAMP CPU bit-identity is covered by the existing metrics tests; real
bf16 speedup is NVIDIA-gated. On Apple this just confirms the bf16 path
compiles and runs finite.

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn/test_phase44_bf16_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU, Tanh
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.ddpg.trainer import DDPGTrainer
from mojo_rl.deep_agents.td3.trainer import TD3Trainer
from mojo_rl.deep_agents.mbpo.trainer import MBPOTrainer
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 64
comptime BATCH = 128
comptime CAP = 20_000
comptime WARMUP = 256
comptime TOTAL = 1_200

comptime DetActor = Sequential[
    Linear[OBS, H], ReLU[H], Linear[H, H], ReLU[H], Linear[H, ACT], Tanh[ACT],
]
comptime StochActor = StochasticActor[
    OBS, ACT, Linear[OBS, H], ReLU[H], Linear[H, H], ReLU[H],
]
comptime Critic = Sequential[
    Linear[OBS + ACT, H], ReLU[H], Linear[H, H], ReLU[H], Linear[H, 1],
]
comptime DynNet = Sequential[
    Linear[OBS + ACT, H], ReLU[H], Linear[H, H], ReLU[H], Linear[H, 2 * (1 + OBS)],
]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def _drive_ddpg(mut env: PendulumEnv[DT]) raises -> Float64:
    var ctx = DeviceContext()
    var tr = DDPGTrainer[
        "gpu", UniformSampleGpuStep[OBS, ACT, BATCH, CAP], DetActor, Critic,
    ].make(
        ctx=ctx, action_scale=Scalar[DT](2.0), learning_starts=WARMUP,
        use_bf16=True,
    )
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    _ = env.reset()
    var o = env.get_obs_list()
    var step = 0
    while step < TOTAL:
        for d in range(OBS):
            obs[d] = o[d]
        tr.select_action(obs, act, step)
        var res = env.step_continuous(act[0])
        var n = res[0].copy()
        for d in range(OBS):
            nxt[d] = n[d]
        tr.record(obs, act, res[1], nxt,
                  Scalar[DT](1.0) if res[2] else Scalar[DT](0.0))
        if res[2]:
            tr.end_episode()
            _ = env.reset()
            o = env.get_obs_list()
        else:
            o = n.copy()
        step += 1
        _ = tr.train_step(step)
    return Float64(tr.flush_metrics().critic_loss.to_f64())


def _drive_td3(mut env: PendulumEnv[DT]) raises -> Float64:
    var ctx = DeviceContext()
    var tr = TD3Trainer[
        "gpu", UniformSampleGpuStep[OBS, ACT, BATCH, CAP], DetActor, Critic,
    ].make(
        ctx=ctx, action_scale=Scalar[DT](2.0), learning_starts=WARMUP,
        use_bf16=True,
    )
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    _ = env.reset()
    var o = env.get_obs_list()
    var step = 0
    while step < TOTAL:
        for d in range(OBS):
            obs[d] = o[d]
        tr.select_action(obs, act, step)
        var res = env.step_continuous(act[0])
        var n = res[0].copy()
        for d in range(OBS):
            nxt[d] = n[d]
        tr.record(obs, act, res[1], nxt,
                  Scalar[DT](1.0) if res[2] else Scalar[DT](0.0))
        if res[2]:
            tr.end_episode()
            _ = env.reset()
            o = env.get_obs_list()
        else:
            o = n.copy()
        step += 1
        _ = tr.train_step(step)
    return Float64(tr.flush_metrics().critic_loss.to_f64())


def _drive_mbpo(mut env: PendulumEnv[DT]) raises -> Float64:
    var ctx = DeviceContext()
    var tr = MBPOTrainer[
        "gpu", StochActor, Critic, DynNet,
        OBS, ACT, BATCH, CAP, 40_000, 3, 2, 10,
    ].make(
        ctx=ctx, action_scale=Scalar[DT](2.0), learning_starts=WARMUP,
        model_train_freq=100, dyn_epochs_per_round=2, num_rollouts_per_step=200,
        sac_updates_per_step=5, dyn_batch_size=128, use_bf16=True,
    )
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    _ = env.reset()
    var o = env.get_obs_list()
    var step = 0
    while step < TOTAL:
        for d in range(OBS):
            obs[d] = o[d]
        tr.select_action(obs, act, step)
        var res = env.step_continuous(act[0])
        var n = res[0].copy()
        for d in range(OBS):
            nxt[d] = n[d]
        tr.record(obs, act, res[1], nxt,
                  Scalar[DT](1.0) if res[2] else Scalar[DT](0.0))
        if res[2]:
            tr.end_episode()
            _ = env.reset()
            o = env.get_obs_list()
        else:
            o = n.copy()
        step += 1
        _ = tr.train_step(step)
    return Float64(tr.flush_metrics().critic_loss.to_f64())


def main() raises:
    print("=" * 70)
    print("Phase 4.4 bf16/AMP GPU smoke (DDPG + TD3 + MBPO)")
    print("=" * 70)
    seed(42)
    var env = PendulumEnv[DT]()

    var ddpg_cl = _drive_ddpg(env)
    print("  DDPG  bf16 critic_loss =", ddpg_cl)
    _finite(ddpg_cl, "ddpg critic_loss")
    assert_true(ddpg_cl >= 0.0, "ddpg critic MSE >= 0")

    var td3_cl = _drive_td3(env)
    print("  TD3   bf16 critic_loss =", td3_cl)
    _finite(td3_cl, "td3 critic_loss")
    assert_true(td3_cl >= 0.0, "td3 critic MSE >= 0")

    var mbpo_cl = _drive_mbpo(env)
    print("  MBPO  bf16 critic_loss =", mbpo_cl)
    _finite(mbpo_cl, "mbpo critic_loss")
    assert_true(mbpo_cl >= 0.0, "mbpo critic MSE >= 0")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

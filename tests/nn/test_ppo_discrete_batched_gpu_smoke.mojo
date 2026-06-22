"""Batched discrete PPO GPU smoke — Craftax-Classic × small N_ENVS.

Functional check of the NEW da2 GPU-batched discrete on-policy path
(`OnPolicyDiscreteAgentBatched` + `run_onpolicy_discrete_train_batched` +
`PPODiscreteAgent.train_batched`) on real hardware via
`BatchedGpuDiscreteEnv` wrapping CraftaxClassicEnv. Tiny config (8 envs,
a handful of updates) — asserts the batched select/record/GAE/update
loop runs finite end-to-end, NOT convergence (the full example is the
serious run). Real convergence is NVIDIA-gated.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_ppo_discrete_batched_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import Tanh
from mojo_rl.deep_agents.ppo_discrete import PPODiscreteAgent
from mojo_rl.deep_agents.training.batched_env import BatchedGpuDiscreteEnv

from mojo_rl.envs.craftax_classic import CraftaxClassicEnv


comptime OBS_DIM = CraftaxClassicEnv[DT].OBS_DIM
comptime NUM_ACTIONS = CraftaxClassicEnv[DT].NUM_ACTIONS
comptime HIDDEN = 64
comptime N_ENVS = 8
comptime ROLLOUT_LEN = 16          # 16*8 = 128 / update
comptime MINIBATCH = 32            # 128 / 32 = 4 minibatches
comptime N_EPOCHS = 2
comptime TOTAL_ENV_STEPS = ROLLOUT_LEN * N_ENVS * 4   # 4 updates

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    print("--- batched discrete PPO GPU smoke (Craftax-Classic x", N_ENVS, ") ---")
    seed(42)
    var ctx = DeviceContext()

    var agent = PPODiscreteAgent[
        "gpu", ActorNet, CriticNet,
        OBS_DIM, NUM_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS, N_ENVS,
    ](
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.01),
    )

    var env = BatchedGpuDiscreteEnv[CraftaxClassicEnv[DT], N_ENVS, OBS_DIM, 1](
        ctx
    )

    _ = agent.train_batched(
        ctx, env, TOTAL_ENV_STEPS, print_every=ROLLOUT_LEN * N_ENVS,
        verbose=True,
    )

    var mean_ret = Float64(agent.mean_return())
    assert_true(not isnan(mean_ret), "mean_return is NaN")
    assert_true(not isinf(mean_ret), "mean_return is Inf")
    print("Final mean ep return (last 10):", mean_ret)
    print("Episodes completed:            ", agent.ep_count())
    print("PASS: batched discrete PPO GPU path runs finite end-to-end.")

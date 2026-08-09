"""PPO GPU Training on Craftax-Classic (deep_agents, GPU-batched).

nn port of the legacy DeepPPOAgent example. Uses the da2 GPU-batched
discrete on-policy path: PPODiscreteAgent.train_batched over a
BatchedGpuDiscreteEnv wrapping CraftaxClassicEnv (256 parallel envs on
device). Feedforward MLP (no recurrence) — same baseline as
Craftax_Baselines `ppo.py`.

Reference (paper / leaderboard, Craftax-Full): PPO 11.9% of max=226
(≈ 2.6 per-episode return at 1B steps); Random ≈ 0. On Classic
(max return ≈ 22) the env/reward/obs/policy-gradient are wired correctly;
this config is sized for a serious GPU run (~16M env steps).

Run with:
    pixi run -e nvidia mojo run -I . examples/craftax_classic/ppo_training_gpu.mojo
    pixi run -e apple  mojo run -I . examples/craftax_classic/ppo_training_gpu.mojo   # slow
"""

from std.random import seed
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.deep_agents.ppo_discrete import PPODiscreteAgent
from mojo_rl.deep_agents.training.batched_env import BatchedGpuDiscreteEnv

from mojo_rl.envs.craftax_classic import CraftaxClassicEnv


comptime OBS_DIM = CraftaxClassicEnv[DT].OBS_DIM       # 1345
comptime NUM_ACTIONS = CraftaxClassicEnv[DT].NUM_ACTIONS  # 17

# Network: wider than classic-control because obs is mostly sparse one-hot.
comptime HIDDEN = 256

# PPO rollout shape — matches Craftax_Baselines defaults closely.
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 256                 # parallel envs on GPU
comptime MINIBATCH = 2048             # 128*256 = 32768 → 16 minibatches
comptime N_EPOCHS = 4

# Each update = ROLLOUT_LEN * N_ENVS = 32,768 transitions. 500 updates
# ≈ 16M env steps. Bump for a publication-grade run.
comptime NUM_UPDATES = 500
comptime TOTAL_ENV_STEPS = ROLLOUT_LEN * N_ENVS * NUM_UPDATES

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO GPU-batched (da2) — Craftax-Classic")
    print("=" * 70)
    print("  obs_dim:", OBS_DIM, " actions:", NUM_ACTIONS, " hidden:", HIDDEN)
    print("  rollout:", ROLLOUT_LEN, " n_envs:", N_ENVS, " minibatch:", MINIBATCH)
    print("  updates:", NUM_UPDATES, " total env steps:", TOTAL_ENV_STEPS)
    print()

    var ctx = DeviceContext()

    var agent = PPODiscreteAgent[
        "gpu", ActorNet, CriticNet,
        OBS_DIM, NUM_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS, N_ENVS,
    ](
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        gae_lambda=Scalar[DT](0.95),
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.01),
        max_grad_norm=Scalar[DT](0.5),
    )

    var env = BatchedGpuDiscreteEnv[CraftaxClassicEnv[DT], N_ENVS, OBS_DIM, 1](
        ctx
    )

    _ = agent.train_batched(
        ctx, env, TOTAL_ENV_STEPS, print_every=ROLLOUT_LEN * N_ENVS * 10,
        verbose=True,
    )

    print("=" * 70)
    print("Final mean ep return (last 10):", agent.mean_return())
    print("Episodes completed:            ", agent.ep_count())
    print("=" * 70)

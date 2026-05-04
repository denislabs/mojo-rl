"""AlphaZero on CartPole — single-player baseline for MuZero comparison.

Uses MCTS-with-true-rules (env as perfect model) to isolate "MCTS + value
learning works on CartPole" from MuZero's learned-dynamics machinery.
A pre-fix MuZero CartPole reportedly hit ≈4-5 reward (worse than random
≈9). If this AZ baseline reaches ~50+ within a few thousand steps,
that's evidence the AZ-style value-learning path is sound and any MuZero
failure on CartPole is in MuZero-specific code (dynamics net, two-hot
encoding, etc.).
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroCartPoleConfig,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== AlphaZero on CartPole ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]
    comptime Config = AlphaZeroCartPoleConfig[
        HIDDEN=64,
        LR=0.001,
        BS=64,
        CAP=20000,
        SIMS=25,
        NODES=64,
        C_PUCT=1.25,
        MAX_EP=500,
    ]

    var agent = GenericAlphaZeroAgent[Config, 16](gamma=0.99)
    print("Agent created:", Config.NAME)

    print("Training (40 iters × 2048 steps × 16 envs = 81920 env steps)...")
    var metrics = agent.train_gpu[CartPoleGPU](
        ctx,
        num_iters=40,
        steps_per_iter=2048,
        train_epochs=4,
        warmup_iters=1,
        verbose=True,
    )
    _ = metrics

    print("=== Done ===")

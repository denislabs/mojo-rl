"""Test MuZero GPU training with GPUDiscreteEnv and config-driven API."""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    print("=== MuZero GPU Test (Config-Driven) ===")

    var ctx = DeviceContext()
    comptime CartPoleGPU = CartPoleEnv[DType.float32]

    comptime Config = MuZeroMLPConfig[
        CartPoleGPU.OBS_DIM,
        CartPoleGPU.NUM_ACTIONS,
        LATENT=32,
        HIDDEN=32,
        BINS=21,
        SIMS=5,
        K=3,
        N=5,
        BS=16,
        CAP=10000,
    ]

    var agent = GenericMuZeroAgent[Config, 16](
        gamma=0.99,
        temperature_decay_steps=5000,
    )
    print("Agent created:", Config.NAME)

    print("Training with n_envs=16 GPU environments...")
    var metrics = agent.train_gpu[CartPoleGPU](
        ctx,
        num_steps=3200,
        warmup_steps=320,
        print_every=1600,
    )

    print("\n=== Results ===")
    print("GPU train steps:", agent.train_step_count)

    if agent.train_step_count > 0:
        print("PASS: GPU training completed")
    else:
        print("FAIL: no training steps")

    _ = metrics
    print("=== Done ===")

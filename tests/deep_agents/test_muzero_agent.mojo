"""Test MuZero agent end-to-end on CartPole with config-driven API."""

from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main():
    print("=== MuZero Agent Test (Config-Driven) ===")

    # Config defines everything: networks, dims, hyperparams
    comptime Config = MuZeroMLPConfig[
        4,
        2,  # obs_dim, action_dim
        LATENT=32,
        HIDDEN=32,
        BINS=21,
        SIMS=5,
        K=3,
        N=5,
        BS=16,
        CAP=10000,
    ]

    var agent = GenericMuZeroAgent[Config](
        gamma=0.99,
        temperature_decay_steps=5000,
    )
    print("Agent created:", Config.NAME)

    var env = CartPoleEnv[DType.float64]()
    print("Training for 2000 steps...")
    var metrics = agent.train[CartPoleEnv[DType.float64]](
        env,
        total_timesteps=2000,
        train_every=4,
        seed_episodes=3,
        print_every=5,
        warmup_steps=200,
    )

    print("\n=== Training Complete ===")
    print("Total train steps:", agent.train_step_count)
    print("Train steps:", agent.train_step_count)

    if agent.train_step_count > 0:
        print("PASS: agent trained successfully")
    else:
        print("FAIL: no training steps executed")

    _ = metrics
    print("=== Done ===")

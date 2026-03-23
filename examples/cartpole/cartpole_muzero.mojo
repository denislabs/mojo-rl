"""CartPole with MuZero — Config-driven model-based RL with MCTS planning.

Usage:
    pixi run mojo run -I . examples/cartpole/cartpole_muzero.mojo
"""

from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.cartpole import CartPoleEnv


def main():
    print("MuZero on CartPole (Config-Driven)")

    comptime Config = MuZeroMLPConfig[
        4, 2, LATENT=128, HIDDEN=128, BINS=51, SIMS=25
    ]

    var agent = GenericMuZeroAgent[Config](
        gamma=0.997,
        v_min=-100.0,
        v_max=100.0,
        temperature=1.0,
        temperature_decay_steps=50000,
    )

    var env = CartPoleEnv[DType.float64]()
    _ = agent.train[CartPoleEnv[DType.float64]](
        env,
        total_timesteps=20000,
        train_every=2,
        seed_episodes=10,
        print_every=50,
        warmup_steps=500,
        use_reanalyze=True,
    )
    print("Done! Train steps:", agent.train_step_count)

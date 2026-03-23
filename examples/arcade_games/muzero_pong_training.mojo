"""Pong with MuZero — Config-driven, clean obs (6D).

Usage:
    pixi run mojo run -I . examples/arcade_games/muzero_pong_training.mojo
"""

from mojo_rl.deep_agents.muzero import GenericMuZeroAgent, MuZeroMLPConfig
from mojo_rl.envs.arcade_games.pong import PongEnv


def main():
    comptime OBS = PongEnv[DType.float64].OBS_DIM  # 6
    comptime ACT = PongEnv[DType.float64].NUM_ACTIONS  # 3

    comptime Config = MuZeroMLPConfig[
        OBS, ACT, LATENT=128, HIDDEN=128, BINS=51, SIMS=25
    ]

    var agent = GenericMuZeroAgent[Config](
        gamma=0.99,
        v_min=-10.0,
        v_max=10.0,
        temperature=1.0,
        temperature_decay_steps=100000,
    )

    var env = PongEnv[DType.float64]()
    _ = agent.train[PongEnv[DType.float64]](
        env,
        total_timesteps=50000,
        train_every=2,
        seed_episodes=20,
        print_every=50,
        warmup_steps=1000,
        use_reanalyze=True,
    )
    print("Done! Train steps:", agent.train_step_count)

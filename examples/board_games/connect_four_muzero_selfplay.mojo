"""ConnectFour self-play training with MuZero/AlphaZero on GPU.

Trains a single network via self-play on Connect Four (7 columns, 6 rows).
Uses GPU MCTS with legal action masking + negated backup.

Usage (Apple Silicon):
    pixi run -e apple mojo run -I . examples/board_games/connect_four_muzero_selfplay.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent
from mojo_rl.deep_agents.muzero.configs import AlphaZeroConfig
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("╔══════════════════════════════════════════════════╗")
    print("║  MuZero Self-Play on Connect Four (GPU)         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    var ctx = DeviceContext()

    comptime C4 = ConnectFourEnv[DType.float32]

    comptime Config = AlphaZeroConfig[
        C4.OBS_DIM,  # 126
        C4.NUM_ACTIONS,  # 7
        HIDDEN=256,
        LR=5e-4,
        BS=128,
        SIMS=50,  # More sims for deeper game
        NODES=128,
    ]

    comptime N_ENVS = 64

    var agent = GenericMuZeroAgent[Config, N_ENVS](
        gamma=1.0,
        v_min=-1.0,
        v_max=1.0,
        temperature=1.0,
        temperature_decay_steps=0,
    )

    print("Config:", Config.NAME)
    print("  Obs:", Config.obs_dim, "| Actions:", Config.action_dim)
    print("  MCTS sims:", Config.num_simulations)
    print("  Envs:", N_ENVS)
    print()

    _ = agent.train_selfplay_gpu[C4](
        ctx,
        num_steps=200_000,
        warmup_steps=2_000,
        gradient_steps=2,
        print_every=20_000,
    )

    print()
    var ckpt_path = "connect_four_muzero.ckpt"
    agent.save_checkpoint(ckpt_path)
    print("Checkpoint saved to:", ckpt_path)
    print("Train steps:", agent.train_step_count)
    print()
    print("Play against the AI:")
    print(
        "  pixi run -e apple mojo run -I ."
        " examples/board_games/connect_four_play_vs_muzero.mojo"
    )

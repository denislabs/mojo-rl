"""Chess self-play demo — random agents play against each other.

Usage:
    pixi run mojo run -I . examples/board_games/chess_self_play.mojo
"""

from mojo_rl.envs.board_games.chess import ChessEnv
from mojo_rl.deep_agents.core.training.self_play_training import (
    self_play_train_cpu,
)


fn main() raises:
    print("=== Chess Self-Play (Random vs Random) ===\n")

    var env = ChessEnv[DType.float64]()
    var metrics = self_play_train_cpu(env, total_games=50, print_every=10)

    print("\nAvg game length:", metrics.mean_steps())
    print("\n=== Done ===")

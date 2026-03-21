"""TicTacToe self-play demo — random agents play against each other.

Validates the self-play infrastructure by running random games
and printing win/draw statistics.

Usage:
    pixi run mojo run -I . examples/board_games/tic_tac_toe_self_play.mojo
"""

from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv
from mojo_rl.deep_agents.core.training.self_play_training import (
    self_play_train_cpu,
)


fn main() raises:
    print("=== TicTacToe Self-Play (Random vs Random) ===\n")

    var env = TicTacToeEnv[DType.float64]()
    var metrics = self_play_train_cpu(env, total_games=10000, print_every=2000)

    print("\nAvg game length:", metrics.mean_steps())
    print("\n=== Done ===")

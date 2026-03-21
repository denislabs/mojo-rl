"""ConnectFour self-play demo — random agents play against each other.

Usage:
    pixi run mojo run -I . examples/board_games/connect_four_self_play.mojo
"""

from mojo_rl.envs.board_games.connect_four import ConnectFourEnv
from mojo_rl.deep_agents.core.training.self_play_training import (
    self_play_train_cpu,
)


fn main() raises:
    print("=== ConnectFour Self-Play (Random vs Random) ===\n")

    var env = ConnectFourEnv[DType.float64]()
    var metrics = self_play_train_cpu(env, total_games=5000, print_every=1000)

    print("\n=== Done ===")

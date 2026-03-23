"""Go 9x9 self-play demo — random agents play against each other.

Usage:
    pixi run mojo run -I . examples/board_games/go_9x9_self_play.mojo
"""

from mojo_rl.envs.board_games.go import GoEnv
from mojo_rl.deep_agents.core.training.self_play_training import (
    self_play_train_cpu,
)


def main() raises:
    print("=== Go 9x9 Self-Play (Random vs Random) ===\n")

    var env = GoEnv[9, DType.float64]()
    var metrics = self_play_train_cpu(env, total_games=100, print_every=20)

    print("\n=== Done ===")

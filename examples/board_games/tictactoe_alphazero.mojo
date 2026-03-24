"""AlphaZero training on TicTacToe — fully GPU.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_alphazero.mojo
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_alphazero.mojo

Then play against the trained agent:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_play_vs_alphazero.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
    AlphaZeroTicTacToeCNNConfig,
    AlphaZeroTicTacToeResNetConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero on TicTacToe ===")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    # Choose architecture:
    # MLP (fastest, decent for TTT):
    # comptime Config = AlphaZeroTicTacToeConfig[]
    # CNN (heavier but better features):
    comptime Config = AlphaZeroTicTacToeCNNConfig[]
    # ResNet (strongest, 50 MCTS sims):
    # comptime Config = AlphaZeroTicTacToeResNetConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    _ = agent.train_selfplay_gpu[TTT, RandomOpponent, GPUMinimaxTicTacToe](
        ctx,
        num_iters=100,
        steps_per_iter=1000,
        train_epochs=10,
        warmup_iters=1,
        arena_threshold=0.5,
        do_eval=True,
        do_eval2=True,
        do_arena=True,
        checkpoint_every=10,
        checkpoint_path="tictactoe_alphazero.ckpt",
    )

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

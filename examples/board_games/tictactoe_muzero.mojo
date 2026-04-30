"""MuZero training on TicTacToe — fully GPU self-play.

MuZero learns a dynamics model g(s,a) and plans in latent space,
unlike AlphaZero which uses true game rules. This tests whether
the learned model is accurate enough for a simple board game.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_muzero.mojo
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== MuZero on TicTacToe ===")
    print()

    comptime Config = MuZeroTicTacToeConfig[]

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64]()

    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 15
    ](
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
        checkpoint_path="tictactoe_muzero.ckpt",
    )

    print()
    print("=== Done ===")

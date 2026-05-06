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

    # temp_threshold=5: temp=1 only for the first 5 moves; greedy from
    # move 6+ during self-play. AlphaZero TTT uses 4. Previously 15 (>9
    # max moves) which made every move exploratory and weakened the
    # endgame policy signal.
    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 5
    ](
        ctx,
        num_iters=100,
        steps_per_iter=1000,
        train_epochs=10,
        warmup_iters=1,
        arena_threshold=0.5,
        do_eval=True,
        do_eval2=True,
        do_arena=False,
        checkpoint_every=10,
        checkpoint_path="tictactoe_muzero.ckpt",
        # Enable GPU reanalyze + Polyak target net (E2 / E4): bootstrap
        # values are refreshed each train step from a slowly-tracking
        # copy of the online networks, mirroring muzero-general's
        # use_last_model_value=True.
        use_reanalyze=True,
    )

    print()
    print("=== Done ===")

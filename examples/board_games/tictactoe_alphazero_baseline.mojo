"""AlphaZero TicTacToe baseline (Phase E5 control run).

Mirrors examples/board_games/tictactoe_muzero.mojo as closely as possible
so we can compare convergence behavior. MLP config (matches MuZero's MLP
architecture), 100 iters x 1000 steps x 10 epochs x 64 envs, no remote logger.
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero TicTacToe baseline (E5 control) ===")
    print()

    comptime Config = AlphaZeroTicTacToeConfig[]

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[Config, 64, 64]()

    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, USE_CUDA_GRAPH=False
    ](
        ctx,
        num_iters=100,
        steps_per_iter=1000,
        train_epochs=10,
        warmup_iters=1,
        arena_threshold=0.55,
        do_eval=True,
        do_eval2=True,
        do_arena=True,
        checkpoint_every=0,  # off for this run
        checkpoint_path="tictactoe_alphazero_baseline.ckpt",
        diag_every=100,
    )

    print()
    print("=== Done ===")

"""AlphaZero — fully GPU batch-then-train.

Like alpha-zero-general: collect → train epochs → eval → arena → repeat.
All on GPU: self-play, training, evaluation, arena.
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
    AlphaZeroTicTacToeCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero — Batch-Then-Train (Fully GPU) ===")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    # Switch between MLP and CNN configs here
    comptime Config = AlphaZeroTicTacToeCNNConfig[]
    # comptime Config = AlphaZeroTicTacToeConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    # Single call — batch-then-train, all on GPU
    _ = agent.train_selfplay_gpu[TTT, RandomOpponent, GPUMinimaxTicTacToe](
        ctx,
        num_iters=250,
        steps_per_iter=1000,  # ~130 games per iter (64 envs × ~7 moves)
        train_epochs=10,  # 10 epochs per iteration
        warmup_iters=1,  # 1 iter warmup (random play)
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

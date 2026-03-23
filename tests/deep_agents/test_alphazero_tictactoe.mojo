"""Quick AlphaZero training test on TicTacToe."""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero TicTacToe Quick Test ===")
    var ctx = DeviceContext()

    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=128, LR=0.01, BS=64, SIMS=200, NODES=256
    ]

    var agent = GenericAlphaZeroAgent[Config, 64]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    print("Before training:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=50)

    # Train 50K steps
    _ = agent.train_selfplay_gpu[TTT](
        ctx,
        num_iters=1,
        steps_per_iter=50000,
        train_epochs=2,
        warmup_iters=1,
        do_eval=False,
        do_arena=False,
    )

    print("\nAfter 50K steps:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=50)

    # Train 50K more
    _ = agent.train_selfplay_gpu[TTT](
        ctx,
        num_iters=1,
        steps_per_iter=50000,
        train_epochs=2,
        do_eval=False,
        do_arena=False,
    )

    print("\nAfter 100K steps:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=50)

    if agent.train_step_count > 0:
        print("\nPASS: AlphaZero trained")
    print("=== Done ===")

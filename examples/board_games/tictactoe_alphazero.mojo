"""TicTacToe AlphaZero — proper self-play training with live evaluation.

Usage:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_alphazero.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent, MinimaxTicTacToe
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


fn main() raises:
    print("╔══════════════════════════════════════════════════╗")
    print("║  AlphaZero on TicTacToe (proper implementation) ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=128, LR=1e-3, BS=128, SIMS=50, NODES=64
    ]
    comptime N_ENVS = 128

    var agent = GenericAlphaZeroAgent[Config, N_ENVS]()

    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    print("Config:", Config.NAME, "| HIDDEN=128 | SIMS=50 | N_ENVS=128")
    print()

    # Initial eval
    print("[Step 0]")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)
    print()

    # Train in chunks
    comptime CHUNK = 50000
    comptime NUM_CHUNKS = 10  # 500K total

    for chunk in range(NUM_CHUNKS):
        _ = agent.train_selfplay_gpu[TTT](
            ctx,
            num_steps=CHUNK,
            warmup_steps=2000 if chunk == 0 else 0,
            gradient_steps=2,
            print_every=CHUNK,
        )

        var step = (chunk + 1) * CHUNK
        print()
        print("[Step", step, "]")
        agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
        agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)
        print()

    print("═══════════════════════════════════════════════════")
    print("Done! Train steps:", agent.train_step_count)
    agent.save_checkpoint("tictactoe_alphazero.ckpt")
    print("Saved: tictactoe_alphazero.ckpt")
    print("═══════════════════════════════════════════════════")

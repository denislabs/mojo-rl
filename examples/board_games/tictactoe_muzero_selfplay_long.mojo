"""TicTacToe self-play — longer training with tuned hyperparams.

Changes vs default:
  - 500K steps (5x more)
  - Larger hidden (256 vs 128)
  - More MCTS sims (50 vs 25)
  - Higher learning rate (3e-3) for faster convergence on small game
  - 2 gradient steps per collection
  - Eval every 50K steps

Usage:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero_selfplay_long.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.deep_agents.muzero.configs import AlphaZeroConfig
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("╔══════════════════════════════════════════════════╗")
    print("║  TicTacToe AlphaZero — Long Run (500K steps)    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    # Tuned config: larger network, more sims, higher LR
    comptime Config = AlphaZeroConfig[
        TTT.OBS_DIM,
        TTT.NUM_ACTIONS,
        HIDDEN=256,  # 256 vs 128
        LR=3e-3,  # Higher LR for small game
        BS=128,  # Larger batch
        SIMS=50,  # More MCTS simulations
        NODES=128,  # More tree nodes
    ]
    comptime N_ENVS = 128  # More parallel games

    var agent = GenericMuZeroAgent[Config, N_ENVS](
        gamma=1.0,
        v_min=-1.0,
        v_max=1.0,
        temperature=1.0,
        temperature_decay_steps=0,
    )

    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    print("Config: HIDDEN=256, LR=3e-3, BS=128, SIMS=50, N_ENVS=128")
    print()

    # Initial eval
    print("[Step 0] Before training:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)
    print()

    # Train in 50K chunks with eval
    comptime CHUNK = 50000
    comptime NUM_CHUNKS = 10  # 500K total

    for chunk in range(NUM_CHUNKS):
        _ = agent.train_selfplay_gpu[TTT](
            ctx,
            num_steps=CHUNK,
            warmup_steps=2000 if chunk == 0 else 0,
            gradient_steps=2,  # 2 gradient steps per collection
            print_every=CHUNK,
        )

        var step = (chunk + 1) * CHUNK
        print()
        print("[Step", step, "] Evaluation (100 games each):")
        agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
        agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)
        print()

    print("═══════════════════════════════════════════════════")
    print("Training complete! Steps:", agent.train_step_count)
    agent.save_checkpoint("tictactoe_muzero.ckpt")
    print("Checkpoint saved.")
    print("═══════════════════════════════════════════════════")

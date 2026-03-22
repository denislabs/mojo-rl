"""TicTacToe self-play training with evaluation during training.

Trains via GPU self-play and periodically evaluates against:
  - Random opponent (should reach ~95%+ win rate)
  - Minimax (perfect play, should reach 100% draw rate)

Usage (Apple Silicon):
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero_selfplay.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.deep_agents.muzero.configs import AlphaZeroConfig
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


fn main() raises:
    print("╔══════════════════════════════════════════════════╗")
    print("║  MuZero Self-Play on TicTacToe (GPU)            ║")
    print("║  With live evaluation vs Random + Minimax        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    comptime Config = AlphaZeroConfig[
        TTT.OBS_DIM, TTT.NUM_ACTIONS,
        HIDDEN=128, LR=1e-3, BS=64, SIMS=25, NODES=64,
    ]
    comptime N_ENVS = 64

    var agent = GenericMuZeroAgent[Config, N_ENVS](
        gamma=1.0, v_min=-1.0, v_max=1.0,
        temperature=1.0, temperature_decay_steps=0,
    )

    # Evaluators
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    print("Training with", N_ENVS, "parallel games...")
    print()

    # Initial evaluation (before training)
    print("[Step 0] Evaluation (before training):")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=50)
    print()

    # Training in chunks with evaluation between each chunk
    comptime CHUNK_SIZE = 20000
    comptime NUM_CHUNKS = 5
    comptime TOTAL = CHUNK_SIZE * NUM_CHUNKS  # 100K

    for chunk in range(NUM_CHUNKS):
        _ = agent.train_selfplay_gpu[TTT](
            ctx,
            num_steps=CHUNK_SIZE,
            warmup_steps=1000 if chunk == 0 else 0,
            gradient_steps=1,
            print_every=CHUNK_SIZE,  # Print once per chunk
        )

        # Sync weights to CPU for evaluation
        # (train_selfplay_gpu already does final sync)

        var step = (chunk + 1) * CHUNK_SIZE
        print()
        print("[Step", step, "] Evaluation:")
        agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)
        agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=50)
        print()

    print("═══════════════════════════════════════════════════")
    print("Training complete! Steps:", agent.train_step_count)

    var ckpt_path = "tictactoe_muzero.ckpt"
    agent.save_checkpoint(ckpt_path)
    print("Checkpoint saved to:", ckpt_path)
    print("═══════════════════════════════════════════════════")
    print()
    print("Play against it:")
    print(
        "  pixi run -e apple mojo run -I ."
        " examples/board_games/tictactoe_play_vs_muzero.mojo"
    )

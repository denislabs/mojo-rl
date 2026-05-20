"""AlphaZero training on TicTacToe — fully GPU with remote logging.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_alphazero.mojo
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_alphazero.mojo

Then play against the trained agent:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_play_vs_alphazero.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
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

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero TicTacToe CNN",
        buffer_size=13,
        api_key=api_key,
    )

    # Choose architecture:
    # CNN smoke test — confirms Conv2DBatchNormReLU + Parallel head code
    # paths work on the known-good TTT setup. BATCH_SIMS=1 matches the
    # MLP-validated proven path (see docs/PHASE_D_GPU_MCTS_BUG_HUNT.md).
    comptime Config = AlphaZeroTicTacToeCNNConfig[
        FILTERS=64,
        LR=0.001,
        BS=64,
        CAP=80000,
        SIMS=50,
        NODES=128,
        C_PUCT=1.0,
        BATCH_SIMS=1,
        VLOSS=3,
    ]
    # MLP (fastest, decent for TTT) — known-converging baseline:
    # comptime Config = AlphaZeroTicTacToeConfig[
    #     HIDDEN=128, LR=0.005, BS=64, CAP=80000, SIMS=50, NODES=128,
    #     C_PUCT=1.0, BATCH_SIMS=1,
    # ]
    # ResNet (strongest, 50 MCTS sims):
    # comptime Config = AlphaZeroTicTacToeResNetConfig[]

    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("history_window", String(Config.history_window))

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[Config, 64, 64, RemoteLogger]()

    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe
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
        # Match CPU example: slow-ramp the replay window so early iters
        # train on recent data only (avoids overfitting iter-1's uniform
        # warmup distribution). Start with last 4 iters, grow by 1 iter
        # every 2 iters until full Config.history_window=20.
        slow_window_start=4,
        slow_window_growth=2,
        checkpoint_every=10,
        checkpoint_path="tictactoe_alphazero.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=500,
        dump_replay=True,
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

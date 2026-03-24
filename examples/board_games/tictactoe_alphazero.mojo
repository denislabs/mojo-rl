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
        run_name="AlphaZero TicTacToe",
        buffer_size=64,
        api_key=api_key,
    )

    # Choose architecture:
    # MLP (fastest, decent for TTT):
    # comptime Config = AlphaZeroTicTacToeConfig[]
    # CNN (heavier but better features):
    comptime Config = AlphaZeroTicTacToeCNNConfig[]
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

    var agent = GenericAlphaZeroAgent[Config, 64, RemoteLogger]()

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
        logger=UnsafePointer(to=logger),
        diag_every=5_000,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

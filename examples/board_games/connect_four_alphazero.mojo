"""AlphaZero training on Connect Four — fully GPU with remote logging.

ResNet architecture with 4 residual blocks, 128 filters, 100 MCTS sims.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero.mojo
    pixi run -e apple mojo run -I . examples/board_games/connect_four_alphazero.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourResNetConfig,
    AlphaZeroConnectFourCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero on Connect Four ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero Connect Four",
        buffer_size=64,
        api_key=api_key,
    )

    # ResNet config (closer to original AlphaZero)
    # comptime Config = AlphaZeroConnectFourResNetConfig[]
    # Lighter CNN alternative:
    comptime Config = AlphaZeroConnectFourCNNConfig[]

    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("max_nodes", String(Config.max_nodes))
    logger.set_config("history_window", String(Config.history_window))

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[Config, 64, RemoteLogger]()

    _ = agent.train_selfplay_gpu[C4, RandomOpponent](
        ctx,
        num_iters=200,
        steps_per_iter=4000,   # More self-play data (C4 games are longer)
        train_epochs=2,        # Minimal overfitting (alpha-zero-general uses 10 with 25 sims)
        warmup_iters=2,        # More warmup for diverse initial data
        arena_threshold=0.55,  # Slightly stricter to avoid accepting noise
        do_eval=True,
        do_arena=True,
        checkpoint_every=10,
        checkpoint_path="connect_four_alphazero.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=5_000,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

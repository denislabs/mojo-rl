"""AlphaZero training on Connect Four — fully GPU with remote logging.

CNN smoke-to-validation: 4× Conv2D+BN+ReLU + FC heads, 128 filters,
SIMS=100, BATCH_SIMS=1 (sequential MCTS — matches the CPU-validated
path; see docs/PHASE_D_GPU_MCTS_BUG_HUNT.md). 30 iters × 2000 steps
with GPUMinimaxConnectFour[5] as the strength oracle. ResNet and
FusedResNet kept as commented fallbacks for later scaling tests.

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
    AlphaZeroConnectFourConfig,
    AlphaZeroConnectFourResNetConfig,
    AlphaZeroConnectFourFusedResNetConfig,
    AlphaZeroConnectFourCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    RandomOpponent,
    GPUMinimaxConnectFour,
)
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
        run_name="AlphaZero ConnectFour CNN",
        buffer_size=13,
        api_key=api_key,
    )

    # CNN — graduated from the validated CPU C4-CNN smoke. BATCH_SIMS=1
    # reuses the CPU-MCTS-equivalent sequential path that fixed TTT's
    # convergence; ResNet/FusedResNet remain as commented fallbacks for
    # later scaling-up tests.
    comptime Config = AlphaZeroConnectFourCNNConfig[
        FILTERS=128,
        LR=2e-3,
        WD=1e-4,
        BS=64,
        CAP=400_000,
        SIMS=100,
        NODES=256,
        C_PUCT=2.0,
        BATCH_SIMS=1,
        VLOSS=3,
    ]
    # MLP (peaked initial policy helps MCTS):
    # comptime Config = AlphaZeroConnectFourConfig[]
    # ResNet (5 residual blocks, closest to original AlphaZero):
    # comptime Config = AlphaZeroConnectFourFusedResNetConfig[NUM_BLOCKS=5]
    # comptime Config = AlphaZeroConnectFourResNetConfig[]

    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("max_nodes", String(Config.max_nodes))
    logger.set_config("history_window", String(Config.history_window))

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[Config, 128, 64, RemoteLogger]()

    _ = agent.train_selfplay_gpu[
        C4,
        RandomOpponent,
        GPUMinimaxConnectFour[5],
    ](
        ctx,
        # Smoke-to-validation budget. The full AlphaZero.jl run uses
        # 50 iters × 110k steps × SIMS=600 — overkill for first GPU
        # CNN validation. Bump steps_per_iter or SIMS once curves show
        # consistent improvement on Minimax-depth-5.
        num_iters=30,
        steps_per_iter=2000,  # ~80 games/iter at ~25 plies avg
        train_epochs=10,
        warmup_iters=1,
        arena_threshold=0.52,  # ~equivalent to avg_reward >= 0.05 (AlphaZero.jl)
        do_eval=True,
        do_eval2=True,  # Eval vs Minimax depth 5 (strength oracle)
        do_arena=True,
        eval_games=32,
        arena_games=40,
        slow_window_start=4,  # Start with 4 iters of history, grow to full
        slow_window_growth=2,  # Grow by 1 every 2 iterations
        checkpoint_every=10,
        checkpoint_path="connect_four_alphazero_cnn.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=500,
        dump_replay=True,
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

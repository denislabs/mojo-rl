"""AlphaZero training on Connect Four — fully GPU with remote logging.

ResNet architecture with 5 residual blocks, 128 filters, 100 MCTS sims.

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
        run_name="AlphaZero Connect Four",
        buffer_size=13,
        api_key=api_key,
    )

    # BATCH_SIMS=6: GenericGPUMCTS requires SIMS % BATCH_SIMS == 0, so
    # we pick 6 (the largest divisor of 600 that's ≤ action_space=7).
    # With 6 ≤ 7, every sim in a round can pick a distinct action — no
    # forced collisions, just within-round Q staleness. At SIMS=600 we
    # get 100 rounds of refinement, well above the ≥20-round threshold.
    # Net win: ~6× MCTS speedup vs BATCH_SIMS=1. If convergence stalls,
    # fall back to BATCH_SIMS=1. See docs/PHASE_D_GPU_MCTS_BUG_HUNT.md.
    #
    # MAX_GRAD_NORM=1.0 = AlphaZero.jl standard; required here because
    # the 5-block ResNet at LR=2e-3 was producing grad_output_norm
    # spikes >100 → policy_ce spikes 1.5→3.0 → arena regression. With
    # clipping, the network can't be shoved through bad regions in one
    # step.
    # MLP config (best for ConnectFour — peaked initial policy helps MCTS)
    # comptime Config = AlphaZeroConnectFourConfig[BATCH_SIMS=6, VLOSS=3]
    # CNN (Conv+BN+ReLU, matching alpha-zero-general):
    # comptime Config = AlphaZeroConnectFourCNNConfig[BATCH_SIMS=6, VLOSS=3]
    # ResNet (closest to original AlphaZero):
    comptime Config = AlphaZeroConnectFourFusedResNetConfig[
        NUM_BLOCKS=5,
        BATCH_SIMS=6,
        VLOSS=3,
        MAX_GRAD_NORM=1.0,
    ]
    # comptime Config = AlphaZeroConnectFourResNetConfig[BATCH_SIMS=6, VLOSS=3]

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
        num_iters=50,
        steps_per_iter=110_000,  # ~5000 games per iter (matching AlphaZero.jl)
        train_epochs=2,  # ~2M sample-updates/iter (2 * 1M / 1024 ≈ 1953 batches, matching AlphaZero.jl)
        warmup_iters=1,
        arena_threshold=0.52,  # ~equivalent to avg_reward >= 0.05 (AlphaZero.jl)
        do_eval=True,
        do_eval2=True,  # Eval vs Minimax depth 5
        do_arena=True,
        eval_games=64,
        arena_games=128,
        slow_window_start=4,  # Start with 4 iters of history, grow to full
        slow_window_growth=2,  # Grow by 1 every 2 iterations
        checkpoint_every=10,
        checkpoint_path="connect_four_alphazero.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=50,
        dump_replay=True,
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

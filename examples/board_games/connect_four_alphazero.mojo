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
    # HISTORY_WINDOW=8 matches AlphaZero.jl steady state: their
    # mem_buffer_size schedule grows 400k → 1M over iter 0 → 15, and
    # at ~125k positions/iter that's ~8 iters of memory at the cap.
    comptime Config = AlphaZeroConnectFourFusedResNetConfig[
        NUM_BLOCKS=5,
        BATCH_SIMS=6,
        VLOSS=3,
        MAX_GRAD_NORM=1.0,
        HISTORY_WINDOW=8,
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
        # AlphaZero.jl-aligned recipe (params.jl):
        #   num_iters=15, num_games=5000, batch_size=1024,
        #   max_batches_per_checkpoint=2000, lr=2e-3, c_puct=2.0,
        #   dirichlet (0.25, 1.0), update_threshold=0.05.
        # Deltas vs AlphaZero.jl noted inline.
        num_iters=15,  # AlphaZero.jl line 64
        steps_per_iter=110_000,  # ~5000 games/iter (AlphaZero.jl line 17)
        # train_epochs=2 → ~1953 batches at 1M-sample buffer, matches
        # AlphaZero.jl's max_batches_per_checkpoint=2000 at steady state.
        # Note: AlphaZero.jl effectively does MORE passes over the
        # smaller early buffer (16 passes at iter 0); without a batch-cap
        # feature we under-train early iters. Open follow-up.
        train_epochs=2,
        warmup_iters=1,
        arena_threshold=0.52,  # AlphaZero.jl update_threshold=0.05 ≈ win_rate ≥ 0.525
        do_eval=True,
        do_eval2=True,
        do_arena=True,
        eval_games=64,
        arena_games=128,  # AlphaZero.jl arena num_games=128
        # AlphaZero.jl's mem_buffer schedule is 400k → 1M over 15 iters,
        # interpolating linearly. With ~125k positions/iter that's ~3
        # iters of memory at iter 0 growing to ~8 iters at iter 15.
        slow_window_start=3,  # ~400k samples at iter 0
        slow_window_growth=3,  # grow by 1 iter every 3 iters → 8 at iter 15
        checkpoint_every=5,  # was 10, but with num_iters=15 we want ≥2 ckpts
        checkpoint_path="connect_four_alphazero.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=50,
        dump_replay=True,
        # use_one_cycle=False matches AlphaZero.jl (plain Adam, no LR
        # schedule — line 53). Was True previously; OneCycle annealing
        # during each iter's pass added an unnecessary moving target on
        # top of the already-changing data distribution.
        use_one_cycle=False,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

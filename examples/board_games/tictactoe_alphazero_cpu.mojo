"""AlphaZero CPU training on TicTacToe (CNN smoke test).

CPU-only — no DeviceContext required. Trains the CNN variant
(Conv2D+BN+ReLU backbone) to optimal TicTacToe play via self-play +
MCTS + supervised policy/value learning. CPU is the convergence oracle
we trust for validating the CNN code paths before GPU.

Target: draw every game against the ``MinimaxTicTacToe`` evaluator (the
perfect-play oracle). At convergence the agent should also draw nearly
all games against itself (P0w/P1w both ≈ 0).

Usage:
    pixi run mojo run -I . examples/board_games/tictactoe_alphazero_cpu.mojo
"""

from std.memory import UnsafePointer
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
    AlphaZeroTicTacToeCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero CPU on TicTacToe (CNN smoke) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero TicTacToe CNN CPU",
        buffer_size=13,
        api_key=api_key,
    )

    # CNN smoke test — mirror of the GPU example so curves are directly
    # comparable on the dashboard. CPU is the convergence oracle.
    #   FILTERS=64  : half the CNN default 128 to keep CPU wall-time
    #                 reasonable; still richer than the MLP backbone.
    #   LR=0.001    : matches the CNN config default (lower than MLP's
    #                 0.005 to avoid dying ReLU in the deeper backbone).
    #   SIMS=50     : same MCTS budget as the MLP-validated baseline so
    #                 only the network changes vs the known-good run.
    #   NODES=128, BS=64, CAP=80000 : as MLP baseline.
    comptime Config = AlphaZeroTicTacToeCNNConfig[
        FILTERS=64,
        LR=0.001,
        BS=64,
        CAP=80000,
        SIMS=50,
        NODES=128,
        C_PUCT=1.0,
    ]
    # MLP baseline (known-converging) — uncomment to revert:
    # comptime Config = AlphaZeroTicTacToeConfig[
    #     HIDDEN=128, LR=0.005, BS=64, CAP=80000, SIMS=50, NODES=128,
    #     C_PUCT=1.0,
    # ]

    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("history_window", String(Config.history_window))
    logger.set_config("device", "cpu")

    var env = TicTacToeEnv[DType.float32]()
    var agent = GenericAlphaZeroAgent[Config, 64, 128, RemoteLogger]()
    var random_opp = RandomOpponent()
    var minimax_opp = MinimaxTicTacToe()

    var t0 = perf_counter_ns()

    _ = agent.train_selfplay_cpu[
        TicTacToeEnv[DType.float32], RandomOpponent, MinimaxTicTacToe
    ](
        env,
        random_opp,
        minimax_opp,
        # Outer loop.
        num_iters=40,
        # ~500 env-steps per iter ≈ 70 self-play games on TTT.
        steps_per_iter=500,
        # 10 epochs over the current replay window per iter.
        train_epochs=10,
        # First iter: uniform-random self-play to seed the buffer.
        warmup_iters=1,
        # Arena: a new model must win ≥55% of decisive games to replace
        # the best. Rejected runs revert params + optimizer state.
        arena_threshold=0.55,
        do_eval=True,
        do_eval2=True,
        do_arena=True,
        eval_games=20,
        arena_games=20,
        # Slow-ramp the replay window: start with the last 4 iterations
        # of history, grow by 1 iter every 2 iters until full
        # ``Config.history_window`` (20).
        slow_window_start=4,
        slow_window_growth=2,
        # Periodic checkpoint so a long run can be resumed.
        checkpoint_every=10,
        checkpoint_path="tictactoe_alphazero_cnn_cpu.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=500,
        verbose=True,
        dump_replay=False,
        # OneCycle LR scaling across each iter's gradient pass.
        use_one_cycle=True,
    )

    var dt_s = Float64(perf_counter_ns() - t0) / 1e9

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("Elapsed:    ", dt_s, "s")
    print("=== Done ===")

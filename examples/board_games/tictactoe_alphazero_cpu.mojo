"""AlphaZero CPU training on TicTacToe (MLP config).

CPU-only — no DeviceContext required. Trains a small MLP to optimal
TicTacToe play via self-play + MCTS + supervised policy/value learning.

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
)
from mojo_rl.deep_agents.muzero.evaluators import (
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero CPU on TicTacToe (MLP) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero TicTacToe CPU",
        buffer_size=13,
        api_key=api_key,
    )

    # Training-grade config — closer to alpha-zero-general defaults.
    #   HIDDEN=128  : two-layer 128-unit MLP for both heads.
    #   SIMS=50     : MCTS sims per move (50 is enough for 3×3 TTT;
    #                 paper uses 100 for boards with larger branching).
    #   NODES=128   : tree node pool; with SIMS=50 ≤ MAX_EP=9 plies
    #                 most search trees stay well below this.
    #   BS=64       : SGD batch size on a CPU is comfortably large.
    #   CAP=80000   : replay capacity; with D4 8× augmentation a single
    #                 iter of 500 env-steps yields ~4k samples, so this
    #                 holds ~20 iters before eviction kicks in.
    #   LR=0.005    : conservative LR; with use_one_cycle we anneal each
    #                 iter's gradient pass through a OneCycle warmup.
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=128,
        LR=0.005,
        BS=64,
        CAP=80000,
        SIMS=50,
        NODES=128,
        C_PUCT=1.0,
    ]

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
        checkpoint_path="tictactoe_alphazero_cpu.ckpt",
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

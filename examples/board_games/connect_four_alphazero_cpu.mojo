"""AlphaZero CPU training on Connect Four (CNN smoke test).

CPU-only — no DeviceContext required. Trains the CNN variant
(Conv2D+BN+ReLU 6×7 backbone) on Connect Four via self-play + MCTS +
supervised policy/value learning. CPU is the convergence oracle we use
to validate CNN code paths before GPU.

Note: no CPU MinimaxConnectFour evaluator exists (only the GPU one), so
``do_eval2`` is off and we only eval vs ``RandomOpponent``. Convergence
signal here is self-play P0w/P1w/Dr balance + arena accept rate + the
training curves (policy_ce, value_mse, target_max_prob) on the
dashboard.

Usage:
    pixi run mojo run -I . examples/board_games/connect_four_alphazero_cpu.mojo
"""

from std.memory import UnsafePointer
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero CPU on Connect Four (CNN smoke) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero ConnectFour CNN CPU",
        buffer_size=13,
        api_key=api_key,
    )

    # CNN smoke config — small enough to run on CPU in reasonable time.
    #   FILTERS=64  : half the CNN default 128 to keep CPU wall-time
    #                 reasonable. Still gives a real Conv backbone.
    #   LR=2e-3, WD=1e-4 : CNN config defaults (AlphaZero.jl).
    #   SIMS=25     : per docstring "proven alpha-zero-general setting"
    #                 for C4 CNN. The default 600 is for full GPU runs.
    #   NODES=64    : sized to SIMS=25.
    #   BS=64, CAP=80000 : smoke-sized replay buffer.
    #   BATCH_SIMS=1, VLOSS=3 : template-required even on CPU (CPU MCTS
    #                 is sequential anyway, so these are inert here).
    comptime Config = AlphaZeroConnectFourCNNConfig[
        FILTERS=64,
        LR=2e-3,
        WD=1e-4,
        BS=64,
        CAP=80000,
        SIMS=25,
        NODES=64,
        C_PUCT=2.0,
        BATCH_SIMS=1,
        VLOSS=3,
    ]

    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("history_window", String(Config.history_window))
    logger.set_config("device", "cpu")

    var env = ConnectFourEnv[DType.float32]()
    var agent = GenericAlphaZeroAgent[Config, 64, 64, RemoteLogger]()
    var random_opp = RandomOpponent()
    # No CPU MinimaxConnectFour — pass RandomOpponent in the eval2 slot
    # and disable do_eval2 below so it never actually runs.
    var fake_minimax = RandomOpponent()

    var t0 = perf_counter_ns()

    _ = agent.train_selfplay_cpu[
        ConnectFourEnv[DType.float32], RandomOpponent, RandomOpponent
    ](
        env,
        random_opp,
        fake_minimax,
        num_iters=10,
        # ~1000 env-steps per iter ≈ 25-40 self-play games on C4 (games
        # average ~25-40 plies). With SIMS=25 each step costs 25 forward
        # passes; total iter time ≈ steps_per_iter × SIMS × CNN_fwd.
        steps_per_iter=1000,
        train_epochs=10,
        warmup_iters=1,
        arena_threshold=0.55,
        do_eval=True,
        do_eval2=False,  # No CPU MinimaxConnectFour available
        do_arena=True,
        eval_games=20,
        arena_games=20,
        slow_window_start=4,
        slow_window_growth=2,
        checkpoint_every=10,
        checkpoint_path="connect_four_alphazero_cnn_cpu.ckpt",
        logger=UnsafePointer(to=logger),
        diag_every=500,
        verbose=True,
        dump_replay=False,
        use_one_cycle=True,
    )

    var dt_s = Float64(perf_counter_ns() - t0) / 1e9

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("Elapsed:    ", dt_s, "s")
    print("=== Done ===")

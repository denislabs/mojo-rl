"""MuZero CPU training on TicTacToe (CNN smoke test).

CPU-only — no DeviceContext required. Trains MuZero (learned dynamics +
MCTS planning) on TicTacToe via self-play. Mirrors the AlphaZero CPU
example so curves are directly comparable on the dashboard:
the same logger keys (policy_ce / value_mse / target_entropy / etc.)
fire from both loops, surfacing trend-direction mismatches that
localize MuZero-specific bugs (à la the AZ Phase D bug hunt — see
docs/PHASE_D_GPU_MCTS_BUG_HUNT.md for the methodology).

Unlike AlphaZero, MuZero learns the dynamics g(s,a)→(r,s') and plans
in latent space, so TicTacToe is the smallest end-to-end test that
exercises all three networks (rep + dyn + pred). CPU is the
convergence oracle for validating CPU↔GPU MuZero parity.

CNN representation (vs MLP baseline): the 2026-05-21 diagnostic
showed D4 augmentation alone fixed the P0/P1 self-play skew but the
MLP rep network was the residual capacity bottleneck (vs-Minimax
never reached 0/20/0). This switches the rep backbone to AZ-style
Conv2D+BN+ReLU stack; dynamics + prediction stay MLP since they
operate on flat latents.

Usage:
    pixi run mojo run -I . examples/board_games/tictactoe_muzero_cpu.mojo
"""

from std.memory import UnsafePointer
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroTicTacToeConfig,
    MuZeroTicTacToeCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== MuZero CPU on TicTacToe (CNN smoke) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero TicTacToe CNN CPU",
        buffer_size=13,
        api_key=api_key,
    )

    # CNN representation backbone. Mirrors AlphaZeroTicTacToeCNNConfig's
    # 4-Conv2DBatchNormReLU stack for the rep network; DYN and PRED
    # stay MLP (operate on flat latents). The 2026-05-21 diagnostic
    # showed the MLP rep was a capacity bottleneck — D4 alone fixed
    # the P0/P1 skew but not the vs-Minimax regression.
    #   FILTERS=64 : half AZ TTT CNN's default 128 to keep CPU wall-time
    #                reasonable; still much richer than the MLP backbone.
    #   LR=0.001   : matches AZ TTT CNN default (deeper backbone with
    #                BN is fine with lower LR than the MLP's 1e-3).
    #   K=3, N=5   : same MCTS / unroll budget as the MLP run for
    #                direct dashboard comparability.
    comptime Config = MuZeroTicTacToeCNNConfig[
        FILTERS=64,
        LATENT=128,
        HIDDEN=128,
        BINS=51,
        LR=1e-3,
        BS=64,
        K=3,
        N=5,
        SIMS=50,
        NODES=128,
        C_PUCT=1.0,
    ]
    # MLP baseline (previous run) — uncomment to revert:
    # comptime Config = MuZeroTicTacToeConfig[
    #     LATENT=64, HIDDEN=64, BINS=51, LR=1e-3, BS=64,
    #     K=3, N=5, SIMS=50, NODES=128, C_PUCT=1.0,
    # ]

    logger.set_config("agent", "MuZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))
    logger.set_config("device", "cpu")

    var env = TicTacToeEnv[DType.float32]()
    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger]()
    var random_opp = RandomOpponent()
    var minimax_opp = MinimaxTicTacToe()

    var t0 = perf_counter_ns()

    _ = agent.train_selfplay_cpu[
        TicTacToeEnv[DType.float32],
        RandomOpponent,
        MinimaxTicTacToe,
        5,  # temp_threshold — comptime, mirrors the GPU example
    ](
        env,
        random_opp,
        minimax_opp,
        num_iters=40,
        # ~500 env-steps per iter ≈ 70 self-play games on TTT.
        steps_per_iter=500,
        # 2 epochs over the current replay window per iter. MuZero
        # GPU uses 2 to avoid the late-iter MCTS-target overfit
        # described in tictactoe_muzero.mojo:67-72.
        train_epochs=2,
        # First iter: uniform-random self-play to seed the buffer.
        warmup_iters=1,
        do_eval=True,
        do_eval2=True,
        eval_games=20,
        # Arena: after each iter, play 20 games new-vs-best via MCTS.
        # Accept if new wins ≥55% of decisive games, else revert all
        # three networks + optimizer state + step counters to the
        # best snapshot. This is what stops the iter-7-onward regression
        # observed in the prior CNN+D4 run (network kept overfitting to
        # self-play distribution after first reaching 0/20/0).
        do_arena=True,
        arena_threshold=0.55,
        arena_games=20,
        # Periodic checkpoint so a long run can be resumed.
        checkpoint_every=10,
        checkpoint_path="tictactoe_muzero_cnn_cpu.ckpt",
        # Reanalyze re-runs MCTS on stored positions with the LATEST
        # networks before a train step (re-freshes mcts_policies +
        # mcts_values targets). With D4 augmentation in place the
        # stored buffer already has 8× the position diversity, so we
        # combine both fixes: data augmentation + gated reanalyze
        # (warmup 500 steps, every 50 steps thereafter — mirrors the
        # EZv2 schedule rather than the naive every-step refresh that
        # amplified perspective bias in the prior diagnostic).
        use_reanalyze=True,
        reanalyze_warmup=500,
        reanalyze_interval=50,
        logger=UnsafePointer(to=logger),
        # Log diag every 500 grad steps. TTT does ~steps_per_epoch ≈
        # buf/BATCH train steps per iter — for a buffer of ~3-4K and
        # BATCH=64 that's ~50 steps × 2 epochs = ~100 per iter, so
        # diag_every=500 gives ~1 datapoint per ~5 iters.
        diag_every=500,
        verbose=True,
        # OneCycle LR scaling across each iter's gradient pass.
        use_one_cycle=True,
    )

    var dt_s = Float64(perf_counter_ns() - t0) / 1e9

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("Elapsed:    ", dt_s, "s")
    print("=== Done ===")

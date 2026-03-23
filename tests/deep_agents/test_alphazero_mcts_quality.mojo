"""Debug: test MCTS policy quality and training convergence.

Diagnoses whether MCTS produces useful policies for TicTacToe.
Tests multiple PUCT values and more training iterations.
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
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
    print("=== MCTS Quality + Training Debug ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    # Low PUCT for deeper search + more sims + more gradient steps
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=128,
        LR=0.01,
        BS=128,
        CAP=200000,  # Larger buffer to prevent forgetting
        SIMS=200,
        NODES=512,
        C_PUCT=0.5,  # Low exploration → deeper search
    ]

    var agent = GenericAlphaZeroAgent[Config, 64]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    print("Config: SIMS=200, NODES=512, C_PUCT=0.5, LR=0.01")
    print()

    # Before training
    print("[0] Before:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)

    # Train in 25K chunks with 8 train epochs per collection
    for chunk in range(10):
        _ = agent.train_selfplay_gpu[TTT](
            ctx,
            num_iters=1,
            steps_per_iter=25000,
            train_epochs=8,
            warmup_iters=1 if chunk == 0 else 0,
            do_eval=False,
            do_arena=False,
        )

        var step = (chunk + 1) * 25000
        print()
        print("[", step, "]")
        agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
        agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)

    # Check replay quality
    print("\nReplay buffer:", agent.state.buf_size, "samples")
    if agent.state.buf_size > 10:
        var pos_count = 0
        var neg_count = 0
        var zero_count = 0
        for i in range(agent.state.buf_size):
            var v = Float64(agent.state.buf_value[i])
            if v > 0.5:
                pos_count += 1
            elif v < -0.5:
                neg_count += 1
            else:
                zero_count += 1
        print("  +1:", pos_count, "| -1:", neg_count, "| 0:", zero_count)

        # Check if policies are sharp (not uniform)
        var sharp_count = 0
        for i in range(min(1000, agent.state.buf_size)):
            var max_p = Float64(0.0)
            for a in range(9):
                var p = Float64(agent.state.buf_policy[i * 9 + a])
                if p > max_p:
                    max_p = p
            if max_p > 0.3:  # More than 30% on one action = somewhat sharp
                sharp_count += 1
        print(
            "  Sharp policies (max>30%):",
            sharp_count,
            "/",
            min(1000, agent.state.buf_size),
        )

    print("\nTrain steps:", agent.train_step_count)
    print("=== Done ===")

"""Sanity check: verify the trained agent produces different actions for different positions."""

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
    print("=== AlphaZero Sanity Check ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=64,
        LR=0.0001,
        BS=64,
        SIMS=200,
        NODES=256,
        C_PUCT=0.5,
    ]

    var agent = GenericAlphaZeroAgent[Config, 32]()

    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    # Train
    print("Training 300K steps...")
    # Train 3 chunks of 100K with eval between each
    for chunk in range(3):
        _ = agent.train_selfplay_gpu[TTT](
            ctx,
            num_iters=1,
            steps_per_iter=100000,
            train_epochs=16,
            warmup_iters=1 if chunk == 0 else 0,
            do_eval=False,
            do_arena=False,
        )
        var step = (chunk + 1) * 100000
        print("[", step, "]")
        agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
        agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)
    print("Train steps:", agent.train_step_count)

    # Test on specific positions
    var env = TTTCPU()

    # Position 1: empty board
    _ = env.reset()
    var legal1 = env.legal_action_mask()
    var obs1 = env.get_obs_list()
    var obs1_f = List[Scalar[dtype]](capacity=27)
    for i in range(27):
        obs1_f.append(
            Scalar[dtype](obs1[i]) if i < len(obs1) else Scalar[dtype](0.0)
        )
    var a1 = agent.select_action(obs1_f, legal1)
    print("\nEmpty board: action =", a1)

    # Position 2: X in center (action 4), now O's turn
    _ = env.reset()
    _ = env._step_impl(4)  # X plays center
    var legal2 = env.legal_action_mask()
    var obs2 = env.get_obs_list()
    var obs2_f = List[Scalar[dtype]](capacity=27)
    for i in range(27):
        obs2_f.append(
            Scalar[dtype](obs2[i]) if i < len(obs2) else Scalar[dtype](0.0)
        )
    var a2 = agent.select_action(obs2_f, legal2)
    print("After X=center: O plays action =", a2, "(legal:", legal2[a2], ")")

    # Position 3: X=center, O=corner, X's turn
    _ = env.reset()
    _ = env._step_impl(4)  # X center
    _ = env._step_impl(0)  # O corner
    var legal3 = env.legal_action_mask()
    var obs3 = env.get_obs_list()
    var obs3_f = List[Scalar[dtype]](capacity=27)
    for i in range(27):
        obs3_f.append(
            Scalar[dtype](obs3[i]) if i < len(obs3) else Scalar[dtype](0.0)
        )
    var a3 = agent.select_action(obs3_f, legal3)
    print("X=center O=corner: X plays action =", a3)

    # All three should be DIFFERENT actions (not always the same)
    if a1 != a2 or a1 != a3 or a2 != a3:
        print("GOOD: Agent produces different actions for different positions")
    else:
        print("BAD: Agent always picks the same action =", a1)

    # Check policy targets in replay — are they diverse?
    print("\nReplay policy analysis (first 20 samples):")
    for s in range(20):
        if s >= agent.state.buf_size:
            break
        var val = Float64(agent.state.buf_value[s])
        print("  s", s, "v=", Int(val), end="")
        for a in range(9):
            var p = Float64(agent.state.buf_policy[s * 9 + a])
            if p > 0.01:
                print(" ", a, ":", Int(p * 100), "%", end="")
        print()

    # Final eval
    print()
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=100)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=100)
    print("=== Done ===")

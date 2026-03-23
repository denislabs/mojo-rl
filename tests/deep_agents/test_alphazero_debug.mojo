"""Debug AlphaZero: check if network learns from replay data."""

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
    print("=== AlphaZero Debug ===")
    var ctx = DeviceContext()

    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=128, LR=0.01, BS=64, SIMS=200, NODES=256
    ]

    var agent = GenericAlphaZeroAgent[Config, 32]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    # Check action selection before training
    var env = TTTCPU()
    _ = env.reset()
    var legal = env.legal_action_mask()
    var obs = env.get_obs_list()
    var obs_f32 = List[Scalar[dtype]](capacity=27)
    for i in range(27):
        if i < len(obs):
            obs_f32.append(Scalar[dtype](obs[i]))
        else:
            obs_f32.append(Scalar[dtype](0.0))

    var action_before = agent.select_action(obs_f32, legal)
    print("Before training, selected action:", action_before)
    # Print first 5 params
    print(
        "  First params:",
        Float64(agent.state.prediction.params[0]),
        Float64(agent.state.prediction.params[1]),
        Float64(agent.state.prediction.params[2]),
    )

    print("\nBefore:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)

    # Train
    print("\nTraining 100K steps...")
    _ = agent.train_selfplay_gpu[TTT](
        ctx,
        num_steps=100000,
        warmup_steps=2000,
        gradient_steps=4,
        print_every=50000,
    )

    # Check replay buffer
    print("\nReplay buffer size:", agent.state.buf_size)
    if agent.state.buf_size > 0:
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
        print(
            "  +1 outcomes:",
            pos_count,
            "| -1 outcomes:",
            neg_count,
            "| 0 outcomes:",
            zero_count,
        )

        # Print first 5 samples
        for s in range(5):
            var val = Float64(agent.state.buf_value[s])
            print("  Sample", s, "value:", val, end="")
            var max_p = Float64(0.0)
            var max_a = 0
            for a in range(9):
                var p = Float64(agent.state.buf_policy[s * 9 + a])
                if p > max_p:
                    max_p = p
                    max_a = a
            print(" best_action:", max_a, "prob:", Int(max_p * 100), "%")

    # Check action selection after training
    var action_after = agent.select_action(obs_f32, legal)
    print("\nAfter training, selected action:", action_after)
    print(
        "  First params:",
        Float64(agent.state.prediction.params[0]),
        Float64(agent.state.prediction.params[1]),
        Float64(agent.state.prediction.params[2]),
    )

    print("\nAfter:")
    agent.print_eval[TTTCPU](eval_env, random_eval, num_games=50)
    agent.print_eval[TTTCPU](eval_env, minimax_eval, num_games=50)

    print("=== Done ===")

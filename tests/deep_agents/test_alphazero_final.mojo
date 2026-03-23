"""AlphaZero — fully GPU training with integrated GPU arena.

All training, self-play, and arena comparison happen on GPU.
CPU evaluation (vs Random/Minimax) only runs occasionally for monitoring.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Linear,
    LinearReLU,
    Sequential,
    Parallel,
    Conv2DReLU,
    FlattenLayer,
)
from mojo_rl.nn.optimizer import Adam
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
    print("=== AlphaZero — Fully GPU Training ===")
    print("Self-play + training + GPU arena (MCTS temp=0)")
    print("CPU evaluation only every 50K steps")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    # Switch between MLP and CNN configs here
    comptime Config = AlphaZeroTicTacToeCNNConfig[]
    # comptime Config = AlphaZeroTicTacToeConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()

    print("Before:")
    var r0 = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
    print("vs Random: W", r0[0], "D", r0[1], "L", r0[2])
    var m0 = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
    print("vs Minimax: W", m0[0], "D", m0[1], "L", m0[2])
    print()

    # Fully integrated GPU training: self-play + training + GPU arena
    # All on GPU, no CPU round-trips during training
    for chunk in range(20):
        _ = agent.train_selfplay_gpu[TTT](
            ctx,
            num_steps=100000,
            warmup_steps=500 if chunk == 0 else 0,
            gradient_steps=4,
            print_every=50000,
            arena_every=25000,  # GPU arena every 25K steps
            arena_games=64,  # n_envs parallel arena games
            arena_threshold=0.6,
        )

        # CPU evaluation only between chunks (every 50K steps)
        var step = (chunk + 1) * 50000
        print("--- Step", step, "---")
        var r = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
        print("vs Random: W", r[0], "D", r[1], "L", r[2])
        var m = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
        print("vs Minimax: W", m[0], "D", m[1], "L", m[2])
        print()

    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

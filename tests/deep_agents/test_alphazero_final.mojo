"""AlphaZero with symmetries + arena comparison + AdamW + step-decay."""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.deep_agents.alphazero import GenericAlphaZeroAgent
from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.deep_agents.muzero.strategies import (
    DirichletNoise, AlphaGoPUCT, SelfPlay,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent, MinimaxTicTacToe
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


struct TTTConfig(AlphaZeroConfig):
    comptime NAME: String = "AlphaZero-TTT-Final"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9
    comptime PredModel = Sequential[
        LinearReLU[27, 128],
        LinearReLU[128, 128],
        Parallel[Linear[128, 9], Linear[128, 1]],
    ]
    comptime OptType = AdamW[LR=0.002, WEIGHT_DECAY=0.01]
    comptime batch_size: Int = 128
    comptime buffer_capacity: Int = 50000
    comptime history_window: Int = 3     # Keep last 3 iterations
    comptime num_simulations: Int = 25    # Like alpha-zero-general
    comptime max_nodes: Int = 64
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]      # cpuct=1 like alpha-zero-general
    comptime Players = SelfPlay


fn main() raises:
    print("=== AlphaZero Final: Symmetries + Arena + AdamW ===")
    print("SIMS=25, cpuct=1.0, LR=0.002/AdamW, WD=0.01")
    print("Arena: every 25K steps, 40 games, 55% threshold")
    print("Symmetries: 8x augmentation for TicTacToe")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    var agent = GenericAlphaZeroAgent[TTTConfig, 64]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()
    var arena_env = TTTCPU()

    print("Before:")
    var r0 = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
    print("vs Random: W", r0[0], "D", r0[1], "L", r0[2])
    var m0 = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
    print("vs Minimax: W", m0[0], "D", m0[1], "L", m0[2])

    for chunk in range(10):
        _ = agent.train_selfplay_gpu[TTT, TTTCPU](
            ctx,
            arena_env,
            num_steps=25000,
            warmup_steps=500 if chunk == 0 else 0,
            gradient_steps=4,
            print_every=25000,
            lr_decay_every=5000,
            lr_decay_factor=0.5,
            arena_every=25000,     # Arena compare every 25K steps
            arena_games=40,
            arena_threshold=0.55,
        )
        var step = (chunk + 1) * 25000
        print("--- Step", step, "---")
        var r = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
        print("vs Random: W", r[0], "D", r[1], "L", r[2])
        var m = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
        print("vs Minimax: W", m[0], "D", m[1], "L", m[2])

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")

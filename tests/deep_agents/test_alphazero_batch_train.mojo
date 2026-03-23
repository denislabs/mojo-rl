"""AlphaZero with batch-then-train approach (like alpha-zero-general).

Instead of continuous stream, this:
1. Collects N complete self-play games
2. Trains for E epochs on ALL collected data
3. Arena compare
4. Repeat

This matches the original AlphaZero training loop structure.
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.alphazero import GenericAlphaZeroAgent
from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.deep_agents.muzero.strategies import (
    DirichletNoise,
    AlphaGoPUCT,
    SelfPlay,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    RandomOpponent,
    MinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


struct TTTBatchConfig(AlphaZeroConfig):
    comptime NAME: String = "AZ-Batch"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9
    comptime PredModel = Sequential[
        LinearReLU[27, 128],
        LinearReLU[128, 128],
        Parallel[Linear[128, 9], Linear[128, 1]],
    ]
    comptime OptType = Adam[LR=0.001]  # Standard Adam, LR=0.001
    comptime batch_size: Int = 64
    comptime buffer_capacity: Int = 50000
    comptime history_window: Int = 3  # Keep last 3 iterations for small game
    comptime num_simulations: Int = 25  # Like alpha-zero-general
    comptime max_nodes: Int = 64
    comptime temp_threshold: Int = 15
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]  # cpuct=1
    comptime Players = SelfPlay


def main() raises:
    print("=== AlphaZero Batch-Then-Train ===")
    print("Collect 100 games → Train 10 epochs → Arena → Repeat")
    print("SIMS=25, cpuct=1, LR=0.001/Adam, 8x symmetries")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    var agent = GenericAlphaZeroAgent[TTTBatchConfig, 64]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()
    var arena_env = TTTCPU()

    print("Before:")
    var r0 = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
    print("vs Random: W", r0[0], "D", r0[1], "L", r0[2])
    var m0 = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
    print("vs Minimax: W", m0[0], "D", m0[1], "L", m0[2])

    # Iteration-based training (like alpha-zero-general)
    for iter in range(20):
        # Mark new iteration — evicts old data beyond history_window
        agent.start_new_iteration()

        # 1. Collect ~100 games via self-play (with 8x symmetries → ~4000 samples)
        # 64 envs × ~7 steps/game ≈ ~450 steps for 64 games
        # Run 1000 env steps to get ~130 games
        _ = agent.train_selfplay_gpu[TTT, TTTCPU](
            ctx,
            arena_env,
            num_steps=1000,
            warmup_steps=0,
            gradient_steps=0,  # No training during collection
            print_every=100000,  # Don't print
        )

        # 2. Train for multiple epochs on collected data
        # With 8x symmetries and ~100 games × ~7 moves = ~5600 samples
        # 10 epochs × 5600/64 batch ≈ 875 gradient steps
        var num_train_steps = 10 * agent.state.buf_size // 64
        if num_train_steps < 10:
            num_train_steps = 10
        if num_train_steps > 2000:
            num_train_steps = 2000

        var gpu = agent.GPUStateType(ctx)
        gpu.upload_from(agent.state, ctx)
        for _ in range(num_train_steps):
            agent.train_step_gpu(ctx, gpu)
        gpu.download_to(agent.state, ctx)

        # 3. Evaluate
        print(
            "Iter",
            iter + 1,
            "(replay:",
            agent.state.buf_size,
            "train:",
            num_train_steps,
            "steps)",
        )
        var r = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
        print("  vs Random: W", r[0], "D", r[1], "L", r[2])
        var m = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
        print("  vs Minimax: W", m[0], "D", m[1], "L", m[2])

    print()
    print("Total train steps:", agent.train_step_count)
    print("=== Done ===")
